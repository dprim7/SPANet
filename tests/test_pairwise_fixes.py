"""Regression tests for the pairwise-attention fixes.

Covers bugs in the on-the-fly ParT-style pairwise implementation:
  1. kinematic recovery must match the LOAD pipeline (log1p only, no z-score).
  2. padding masked with +1.0 instead of -inf (padded jets leaked into attention).
  3. BatchNorm computed over zeroed padded pairs, skewing the statistics.
  4. delta_phi must wrap on both sides (fmod kept the dividend sign).

These only need torch + spanet.network.jet_reconstruction_pairwise.pairwise_features
(which has no heavy deps), so they run without a dataset or Options.

Run:  PYTHONPATH=. python tests/test_pairwise_fixes.py
"""
import math
import torch

from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    MaskedBatchNorm1d,
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    delta_phi,
    extract_physical_kinematics,
)


def test_delta_phi_wraps_below_minus_pi():
    """delta_phi must wrap into [-pi, pi) on BOTH sides.

    The old torch.fmod version kept the dividend's sign: for dphi < -pi the
    shifted value was negative and came back unwrapped (|dphi| in (pi, 2pi)).
    """
    # Failing branch: phi1=-3, phi2=3 -> raw dphi = -6 -> wrapped ~ +0.2832.
    d = delta_phi(torch.tensor([-3.0]), torch.tensor([3.0]))
    expected = -6.0 + 2 * math.pi
    assert abs(d.item() - expected) < 1e-6, d.item()

    # Mirrored control (this branch worked even with fmod): +6 -> 6 - 2pi.
    d = delta_phi(torch.tensor([3.0]), torch.tensor([-3.0]))
    assert abs(d.item() - (6.0 - 2 * math.pi)) < 1e-6, d.item()

    # Property sweep against the %-based reference over a phi grid.
    grid = torch.linspace(-math.pi, math.pi, 41)
    p1, p2 = torch.meshgrid(grid, grid, indexing="ij")
    d = delta_phi(p1, p2)
    assert (d.abs() <= math.pi + 1e-6).all(), d.abs().max()
    ref = (p1 - p2 + math.pi) % (2 * math.pi) - math.pi
    assert torch.allclose(d, ref, atol=1e-6)
    print("OK delta_phi wraps below -pi")


def test_kinematic_recovery_matches_dataset_pipeline():
    """extract_physical_kinematics must invert the LOAD-time transform exactly.

    The dataset pipeline (SequentialInput.load) applies ONLY log(x+1) to
    log_scale features at load; z-scoring happens later, out-of-place, inside
    CombinedVectorEmbedding via Normalizer and never mutates the raw sources.
    So stored = log1p(pt) -- NOT (log1p(pt) - mean)/std. This test builds
    source_data exactly as the loader does and asserts exact physical
    recovery; an earlier revision applied x*std+mean here (mis-modeling the
    pipeline) and its test was vacuous (it tested its own assumption without
    calling production code).
    """
    torch.manual_seed(0)
    B, N = 3, 6
    pt_phys = torch.rand(B, N) * 1980.0 + 20.0   # 20..2000 GeV
    eta_phys = torch.randn(B, N)
    phi_phys = torch.rand(B, N) * 2 * math.pi - math.pi
    mass_phys = torch.rand(B, N) * 150.0

    # Build source_data exactly as SequentialInput.load does: log1p for
    # log_scale features, raw otherwise, NO z-score.
    cols = {
        "pt": torch.log1p(pt_phys),        # pt: log_normalize -> log1p stored
        "eta": eta_phys,                   # eta: normalize -> raw stored
        "sinphi": torch.sin(phi_phys),
        "cosphi": torch.cos(phi_phys),
        "mass": mass_phys,                 # v11 mass: normalize -> raw stored
    }
    source_data = torch.stack(list(cols.values()), dim=-1)
    kin = {
        "pt_idx": 0, "eta_idx": 1, "sinphi_idx": 2, "cosphi_idx": 3,
        "phi_idx": -1, "mass_idx": 4, "use_sincos_phi": True,
    }

    pt, eta, phi, mass = extract_physical_kinematics(
        source_data, kin, pt_is_log=True, mass_is_log=False
    )
    assert torch.allclose(pt, pt_phys, atol=1e-3), (pt - pt_phys).abs().max()
    assert torch.allclose(eta, eta_phys, atol=1e-6)
    assert torch.allclose(phi, phi_phys, atol=1e-5)
    assert torch.allclose(mass, mass_phys, atol=1e-6)

    # log-scaled mass variant (e.g. upstream ttbar event files).
    source_log_mass = source_data.clone()
    source_log_mass[:, :, 4] = torch.log1p(mass_phys)
    _, _, _, mass2 = extract_physical_kinematics(
        source_log_mass, kin, pt_is_log=True, mass_is_log=True
    )
    assert torch.allclose(mass2, mass_phys, atol=1e-3)

    # Control: the old recipe (x*std + mean before expm1, with stats of the
    # log1p'd values) does NOT recover physical pt from load-pipeline tensors.
    stored_pt = cols["pt"]
    mean, std = stored_pt.mean(), stored_pt.std()
    old_recipe = torch.expm1(stored_pt * std + mean).clamp(min=0)
    assert not torch.allclose(old_recipe, pt_phys, rtol=0.1), \
        "old x*std+mean recipe unexpectedly recovered physical pt"
    print("OK kinematic recovery matches dataset pipeline")


def test_fix3_masked_batchnorm_ignores_padding():
    """MaskedBatchNorm1d stats must be computed over valid positions only."""
    torch.manual_seed(0)
    B, C, L = 4, 3, 20
    x = torch.randn(B, C, L) * 2.0 + 5.0
    mask = torch.ones(B, 1, L)
    mask[:, :, L // 2:] = 0.0
    x = x * mask  # padded entries are zeroed, exactly as in the real pipeline

    mbn = MaskedBatchNorm1d(C).train()
    y = mbn(x, mask)
    valid = mask.bool().expand(B, C, L)
    ym = y[valid]
    assert abs(ym.mean().item()) < 1e-4, ym.mean().item()
    assert abs(ym.std(unbiased=False).item() - 1.0) < 1e-2, ym.std().item()

    # Without the mask the zeros drag the statistics off -> valid entries not standardized.
    mbn2 = MaskedBatchNorm1d(C).train()
    y_unmasked = mbn2(x, None)
    assert abs(y_unmasked[valid].mean().item()) > 1e-2
    print("OK fix#3 masked batchnorm")


def test_fix3_embedding_padded_pairs_do_not_affect_valid_bias():
    """Changing padded-pair inputs must not change the bias on valid pairs."""
    torch.manual_seed(0)
    B, N, F, H = 2, 6, 4, 4
    feats = torch.randn(B, N, N, F)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, 4:] = False  # last two particles padded
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    i = torch.arange(N)
    pair_mask = pair_mask.clone()
    pair_mask[:, i, i] = False

    emb = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8).eval()
    feats_masked = feats * pair_mask.unsqueeze(-1)
    bias1 = emb(feats_masked, pair_mask)

    feats2 = feats.clone()
    feats2[:, 4:, :, :] += 50.0  # perturb padded-particle rows
    feats2[:, :, 4:, :] += 50.0
    feats2_masked = feats2 * pair_mask.unsqueeze(-1)
    bias2 = emb(feats2_masked, pair_mask)

    bias1 = bias1.view(B, H, N, N)
    bias2 = bias2.view(B, H, N, N)
    valid_block = bias1[:, :, :4, :4]
    assert torch.allclose(valid_block, bias2[:, :, :4, :4], atol=1e-5)
    print("OK fix#3 embedding padded-pair invariance")


def test_feature_computer_returns_mask_and_zeros_selfpairs():
    """New return signature: (features, pair_mask); diagonal excluded/zeroed."""
    torch.manual_seed(0)
    B, N = 2, 5
    pt = torch.rand(B, N) * 100
    eta = torch.randn(B, N)
    phi = torch.rand(B, N) * 2 * math.pi - math.pi
    mass = torch.rand(B, N) * 20
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 4:] = False

    comp = PairwiseFeatureComputer(num_features=4)
    feats, pair_mask = comp(pt, eta, phi, mass, mask)
    assert feats.shape == (B, N, N, 4)
    i = torch.arange(N)
    assert torch.count_nonzero(feats[:, i, i, :]) == 0  # self-pairs zeroed
    assert not pair_mask[:, i, i].any()                 # diagonal excluded
    assert not pair_mask[:, 4, :].any()                 # padded rows excluded
    print("OK feature computer signature + masking")


def test_fix2_neg_inf_masks_padding_in_attention():
    """The -inf additive key_padding_mask excludes padded keys; +1.0 does not.

    Isolates the masking logic from the fix using a raw MultiheadAttention, so
    the test has no Options/GRU dependency.
    """
    torch.manual_seed(0)
    T, Bt, E, Hh = 5, 2, 8, 2
    mha = torch.nn.MultiheadAttention(E, Hh, dropout=0.0).eval()
    x = torch.randn(T, Bt, E)
    bias = torch.zeros(Bt * Hh, T, T)  # float attn_mask (pairwise bias placeholder)

    n_valid = 3
    padding = torch.zeros(Bt, T, dtype=torch.bool)
    padding[:, n_valid:] = True  # last two positions padded

    def run(kpm, xin):
        out, _ = mha(xin, xin, xin, key_padding_mask=kpm, attn_mask=bias, need_weights=False)
        return out[:n_valid]  # only VALID query rows matter (padded rows are masked downstream)

    # Perturb the padded positions (their value vectors). Valid query rows must not
    # change if padded KEYS are properly excluded from attention.
    x_pert = x.clone(); x_pert[n_valid:] += 100.0

    # Correct fix: additive -inf mask -> padded keys excluded -> valid rows invariant.
    kpm_neg_inf = torch.zeros_like(padding, dtype=torch.float32).masked_fill(padding, float("-inf"))
    d_fixed = (run(kpm_neg_inf, x) - run(kpm_neg_inf, x_pert)).abs().max().item()
    assert d_fixed < 1e-4, f"padded keys leaked into valid rows with -inf mask: {d_fixed}"

    # Old bug: +1.0 mask -> padded keys still attended -> valid rows change.
    kpm_plus_one = padding.float()  # True -> 1.0
    d_bug = (run(kpm_plus_one, x) - run(kpm_plus_one, x_pert)).abs().max().item()
    assert d_bug > 1e-3, "expected +1.0 mask to leak into valid rows, but it did not"
    print("OK fix#2 -inf masks padding (and +1.0 leaks)")


if __name__ == "__main__":
    test_delta_phi_wraps_below_minus_pi()
    test_kinematic_recovery_matches_dataset_pipeline()
    test_fix2_neg_inf_masks_padding_in_attention()
    test_fix3_masked_batchnorm_ignores_padding()
    test_fix3_embedding_padded_pairs_do_not_affect_valid_bias()
    test_feature_computer_returns_mask_and_zeros_selfpairs()
    print("\nAll pairwise fix tests passed.")
