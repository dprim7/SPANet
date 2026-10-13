"""Tests for multi-collection pairwise attention.

Covers the extension of the pairwise bias from a single collection to the full
concatenated sequence of jet collections (intra- AND cross-collection pairs):
  - prefixed kinematic detection (fj_pt / vfj_pt / ... ; sdmass NOT mass)
  - alignment of combined kinematics with the embedding concatenation order
  - physically-correct cross-collection features (deltaR between collections)
  - non-participating segments are masked out
  - bias is exactly zero on invalid pairs

Torch-only; no dataset or Options needed.
Run:  PYTHONPATH=. python tests/test_pairwise_multicollection.py
"""
import math
import torch

from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    combine_collection_kinematics,
)


def test_prefixed_detection_boosted():
    names = ["fj_pt", "fj_eta", "fj_sinphi", "fj_cosphi", "fj_mass", "fj_sdmass", "fj_tau21"]
    kin = auto_detect_kinematic_features(names, "BoostedJets")
    assert kin["pt_idx"] == 0
    assert kin["eta_idx"] == 1
    assert kin["use_sincos_phi"] and kin["sinphi_idx"] == 2 and kin["cosphi_idx"] == 3
    assert kin["mass_idx"] == 4, "must pick fj_mass, NOT fj_sdmass"
    print("OK prefixed detection (fj_)")


def test_prefixed_detection_veryboosted_and_bare():
    names = ["vfj_pt", "vfj_eta", "vfj_sinphi", "vfj_cosphi", "vfj_mass", "vfj_sdmass"]
    kin = auto_detect_kinematic_features(names, "VeryBoostedJets")
    assert kin["pt_idx"] == 0 and kin["eta_idx"] == 1 and kin["mass_idx"] == 4
    assert kin["use_sincos_phi"]

    # Bare names must keep working exactly as before.
    names = ["pt", "eta", "sinphi", "cosphi", "btag", "mass"]
    kin = auto_detect_kinematic_features(names, "Jets")
    assert kin["pt_idx"] == 0 and kin["eta_idx"] == 1 and kin["mass_idx"] == 5
    assert kin["use_sincos_phi"] and kin["sinphi_idx"] == 2 and kin["cosphi_idx"] == 3
    print("OK prefixed detection (vfj_ + bare)")


def test_combine_alignment_and_cross_collection_deltaR():
    # Collection A: 2 small-R jets; collection B: 1 boosted jet.
    ptA = torch.tensor([[100.0, 50.0]])
    etaA = torch.tensor([[0.0, 1.0]])
    phiA = torch.tensor([[0.0, 0.5]])
    mA = torch.tensor([[10.0, 5.0]])
    maskA = torch.ones(1, 2, dtype=torch.bool)

    ptB = torch.tensor([[300.0]])
    etaB = torch.tensor([[-1.0]])
    phiB = torch.tensor([[2.0]])
    mB = torch.tensor([[80.0]])
    maskB = torch.ones(1, 1, dtype=torch.bool)

    pt, eta, phi, mass, mask = combine_collection_kinematics([
        (2, (ptA, etaA, phiA, mA, maskA)),
        (1, (ptB, etaB, phiB, mB, maskB)),
    ])
    # Alignment: positions 0-1 = collection A, position 2 = collection B.
    assert pt.shape == (1, 3)
    assert pt[0, 0] == 100.0 and pt[0, 1] == 50.0 and pt[0, 2] == 300.0
    assert mass[0, 2] == 80.0 and mask.all()

    # Cross-collection deltaR (jet 0 <-> boosted jet) must match manual physics.
    comp = PairwiseFeatureComputer(num_features=3)
    feats, pair_mask = comp(pt, eta, phi, mass, mask)
    d_eta = etaA[0, 0] - etaB[0, 0]                     # 0 - (-1) = 1
    d_phi = (phiA[0, 0] - phiB[0, 0] + math.pi) % (2 * math.pi) - math.pi  # -2
    dr = math.sqrt(float(d_eta) ** 2 + float(d_phi) ** 2)
    ln_dr = feats[0, 0, 2, 2].item()                    # feature index 2 = ln(deltaR)
    assert abs(ln_dr - math.log(dr)) < 1e-3, (ln_dr, math.log(dr))
    # Symmetry of deltaR and validity of the cross pair.
    assert abs(feats[0, 2, 0, 2].item() - ln_dr) < 1e-6
    assert pair_mask[0, 0, 2] and pair_mask[0, 2, 0]
    print("OK combine alignment + cross-collection deltaR")


def test_non_participating_segment_masked():
    ptA = torch.tensor([[100.0, 50.0]])
    etaA = torch.zeros(1, 2)
    phiA = torch.zeros(1, 2)
    maskA = torch.ones(1, 2, dtype=torch.bool)

    pt, eta, phi, mass, mask = combine_collection_kinematics([
        (2, (ptA, etaA, phiA, None, maskA)),
        (3, None),  # non-participating segment (e.g. a global input)
    ])
    assert pt.shape == (1, 5)
    assert not mask[0, 2:].any(), "non-participating positions must be masked out"
    assert mass[0, :2].sum() == 0.0  # mass=None -> zeros

    comp = PairwiseFeatureComputer(num_features=4)
    feats, pair_mask = comp(pt, eta, phi, mass, mask)
    assert not pair_mask[0, 2:, :].any() and not pair_mask[0, :, 2:].any()
    assert torch.count_nonzero(feats[0, 2:, :, :]) == 0
    print("OK non-participating segment masked")


def test_bias_exactly_zero_on_invalid_pairs():
    torch.manual_seed(0)
    B, N, F, H = 2, 6, 4, 4
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, 4:] = False
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    i = torch.arange(N)
    pair_mask = pair_mask.clone()
    pair_mask[:, i, i] = False

    feats = torch.randn(B, N, N, F) * pair_mask.unsqueeze(-1)
    emb = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8).eval()
    bias = emb(feats, pair_mask).view(B, H, N, N)

    invalid = ~pair_mask
    assert torch.count_nonzero(bias[invalid.unsqueeze(1).expand(B, H, N, N)]) == 0, (
        "bias must be exactly zero on padded / self / non-participating pairs"
    )
    assert torch.count_nonzero(bias[:, :, 0, 1]) > 0, "valid pairs should have nonzero bias"
    print("OK bias exactly zero on invalid pairs")


if __name__ == "__main__":
    test_prefixed_detection_boosted()
    test_prefixed_detection_veryboosted_and_bare()
    test_combine_alignment_and_cross_collection_deltaR()
    test_non_participating_segment_masked()
    test_bias_exactly_zero_on_invalid_pairs()
    print("\nAll multi-collection pairwise tests passed.")
