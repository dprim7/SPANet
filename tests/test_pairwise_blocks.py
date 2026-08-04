"""Tests for block-structured pairwise embeddings (per-pair-type MLPs).

Covers the rectangular cross-type path and the per-block masked-BN structure
adopted from the cross_source design, built on the fixed foundation
(pt denorm, -inf padding, masked BatchNorm, bias zeroing).

Torch-only; no dataset or Options needed.
Run:  PYTHONPATH=. python tests/test_pairwise_blocks.py
"""
import math
import torch

from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
)


def _kin(pt, eta, phi, mass=None):
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return (t(pt), t(eta), t(phi), None if mass is None else t(mass))


def test_rectangular_cross_block_physics():
    """Cross-type (rectangular) features must match manual physics."""
    comp = PairwiseFeatureComputer(num_features=4)
    # 2 small-R jets vs 1 boosted jet
    kin_a = _kin([100.0, 50.0], [0.0, 1.0], [0.0, 0.5], [10.0, 5.0])
    kin_b = _kin([300.0], [-1.0], [2.0], [80.0])
    mask_a = torch.ones(1, 2, dtype=torch.bool)
    mask_b = torch.ones(1, 1, dtype=torch.bool)

    feats, pmask = comp.compute_block(kin_a, kin_b, mask_a, mask_b, remove_self_pair=False)
    assert feats.shape == (1, 2, 1, 4)
    assert pmask.shape == (1, 2, 1) and pmask.all()

    d_eta = 0.0 - (-1.0)
    # %-based wrap follows the divisor sign (correct for dphi < -pi);
    # math.fmod would reproduce the pre-fix delta_phi bug.
    d_phi = (0.0 - 2.0 + math.pi) % (2 * math.pi) - math.pi
    dr = math.sqrt(d_eta ** 2 + d_phi ** 2)
    assert abs(feats[0, 0, 0, 2].item() - math.log(dr)) < 1e-3   # ln(deltaR)
    kt = min(100.0, 300.0) * dr
    assert abs(feats[0, 0, 0, 0].item() - math.log(kt)) < 1e-3   # ln(kT)

    # Swapping i and j must give identical features (symmetry).
    feats_T, _ = comp.compute_block(kin_b, kin_a, mask_b, mask_a, remove_self_pair=False)
    assert torch.allclose(feats_T[0, 0, :, :], feats[0, :, 0, :], atol=1e-6)
    print("OK rectangular cross-block physics + symmetry")


def test_self_pair_removed_only_on_square():
    comp = PairwiseFeatureComputer(num_features=4)
    kin = _kin([100.0, 50.0], [0.0, 1.0], [0.0, 0.5], [10.0, 5.0])
    mask = torch.ones(1, 2, dtype=torch.bool)

    feats_sq, pmask_sq = comp.compute_block(kin, kin, mask, mask, remove_self_pair=True)
    i = torch.arange(2)
    assert torch.count_nonzero(feats_sq[:, i, i, :]) == 0
    assert not pmask_sq[:, i, i].any()

    # rectangular block: no diagonal concept, nothing removed
    kin_b = _kin([300.0], [-1.0], [2.0], [80.0])
    mask_b = torch.ones(1, 1, dtype=torch.bool)
    feats_r, pmask_r = comp.compute_block(kin, kin_b, mask, mask_b, remove_self_pair=False)
    assert pmask_r.all()
    print("OK self-pair removal only on square blocks")


def test_padding_masked_in_blocks():
    comp = PairwiseFeatureComputer(num_features=4)
    kin_a = _kin([100.0, 50.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.0])
    mask_a = torch.tensor([[True, True, False]])
    kin_b = _kin([300.0, 0.0], [-1.0, 0.0], [2.0, 0.0])
    mask_b = torch.tensor([[True, False]])

    feats, pmask = comp.compute_block(kin_a, kin_b, mask_a, mask_b)
    assert pmask.shape == (1, 3, 2)
    assert not pmask[0, 2, :].any() and not pmask[0, :, 1].any()
    assert torch.count_nonzero(feats[0, 2, :, :]) == 0
    assert torch.count_nonzero(feats[0, :, 1, :]) == 0
    print("OK padding masked in rectangular blocks")


def test_rectangular_embedding_and_bias_zeroing():
    torch.manual_seed(0)
    B, R, C, F, H = 2, 5, 3, 4, 4
    emb = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8).eval()
    pmask = torch.ones(B, R, C, dtype=torch.bool)
    pmask[:, 4, :] = False        # padded row
    pmask[:, :, 2] = False        # padded column
    feats = torch.randn(B, R, C, F) * pmask.unsqueeze(-1)

    bias = emb(feats, pmask)
    assert bias.shape == (B * H, R, C)
    b = bias.view(B, H, R, C)
    assert torch.count_nonzero(b[:, :, 4, :]) == 0
    assert torch.count_nonzero(b[:, :, :, 2]) == 0
    assert torch.count_nonzero(b[:, :, 0, 0]) > 0
    print("OK rectangular embedding + bias zeroing")


def test_per_block_bn_independence():
    """Two blocks with wildly different scales must not share statistics."""
    torch.manual_seed(0)
    F, H = 4, 2
    emb_small = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8).train()
    emb_large = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8).train()

    small = torch.randn(8, 6, 6, F) * 1.0 + 4.0     # e.g. Jet-Jet ln kT scale
    large = torch.randn(8, 3, 3, F) * 2.0 + 12.0    # e.g. Boosted-Boosted scale
    m_s = torch.ones(8, 6, 6, dtype=torch.bool)
    m_l = torch.ones(8, 3, 3, dtype=torch.bool)

    emb_small(small, m_s)
    emb_large(large, m_l)

    mu_s = emb_small.input_bn.running_mean.mean().item()
    mu_l = emb_large.input_bn.running_mean.mean().item()
    # Each block's BN tracked its own population, not a pooled mixture.
    assert mu_s < mu_l, (mu_s, mu_l)
    assert abs(mu_s - 0.4) < 0.2 and abs(mu_l - 1.2) < 0.4  # momentum=0.1 of true means
    print("OK per-block BN statistics are independent")


def test_block_assembly_pattern():
    """Full-matrix assembly: blocks at offsets, transpose mirror, zeros elsewhere."""
    torch.manual_seed(0)
    F, H, B = 4, 2, 1
    comp = PairwiseFeatureComputer(num_features=F)
    emb_aa = PairwiseEmbedding(F, H, 8).eval()
    emb_bb = PairwiseEmbedding(F, H, 8).eval()
    emb_ab = PairwiseEmbedding(F, H, 8).eval()

    kin_a = _kin([100.0, 50.0], [0.0, 1.0], [0.0, 0.5], [10.0, 5.0])
    kin_b = _kin([300.0], [-1.0], [2.0], [80.0])
    mask_a = torch.ones(B, 2, dtype=torch.bool)
    mask_b = torch.ones(B, 1, dtype=torch.bool)
    N = 3  # 2 jets + 1 boosted

    bias = torch.zeros(B * H, N, N)
    f_aa, m_aa = comp.compute_block(kin_a, kin_a, mask_a, mask_a, remove_self_pair=True)
    bias[:, 0:2, 0:2] = emb_aa(f_aa, m_aa)
    f_bb, m_bb = comp.compute_block(kin_b, kin_b, mask_b, mask_b, remove_self_pair=True)
    bias[:, 2:3, 2:3] = emb_bb(f_bb, m_bb)
    f_ab, m_ab = comp.compute_block(kin_a, kin_b, mask_a, mask_b)
    blk = emb_ab(f_ab, m_ab)
    bias[:, 0:2, 2:3] = blk
    bias[:, 2:3, 0:2] = blk.transpose(1, 2)

    # cross block symmetric under transpose
    assert torch.allclose(bias[:, 0:2, 2:3], bias[:, 2:3, 0:2].transpose(1, 2))
    # diagonal (self-pairs) zero
    for k in range(N):
        assert torch.count_nonzero(bias[:, k, k]) == 0
    # off-diagonal valid entries nonzero
    assert torch.count_nonzero(bias[:, 0, 1]) > 0 and torch.count_nonzero(bias[:, 0, 2]) > 0
    print("OK block assembly pattern (offsets, transpose, zeros)")


if __name__ == "__main__":
    test_rectangular_cross_block_physics()
    test_self_pair_removed_only_on_square()
    test_padding_masked_in_blocks()
    test_rectangular_embedding_and_bias_zeroing()
    test_per_block_bn_independence()
    test_block_assembly_pattern()
    print("\nAll pairwise block tests passed.")
