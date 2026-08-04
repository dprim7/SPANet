"""Tests for the configurable pairwise feature set and the Minkowski channel.

The "mink" channel is the PELICAN-style Lorentz invariant
    ln(2 p_i.p_j) = ln(m2_ij - m_i^2 - m_j^2)
i.e. the pair mass with the self-masses removed -- checked against both forms.

Torch-only, no dataset or Options needed.

Run:  PYTHONPATH=. python tests/test_pairwise_minkowski.py
"""
import math
import torch

from spanet.network.jet_reconstruction_pairwise.pairwise_features import (
    CANONICAL_PAIRWISE_FEATURES,
    PairwiseEmbedding,
    PairwiseFeatureComputer,
    parse_pairwise_feature_set,
)


def _kin(pt, eta, phi, mass=None):
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    return (t(pt), t(eta), t(phi), None if mass is None else t(mass))


def _four_vector(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt((pt * math.cosh(eta)) ** 2 + mass ** 2)
    return px, py, pz, e


def test_minkowski_dot_identity():
    """mink channel must equal ln(2 p_i.p_j) AND ln(m2_ij - m_i^2 - m_j^2)."""
    comp = PairwiseFeatureComputer(feature_names=("m2", "mink"))
    kin_a = _kin([100.0, 50.0], [0.0, 1.0], [0.0, 0.5], [10.0, 5.0])
    kin_b = _kin([300.0], [-1.0], [2.0], [80.0])
    mask_a = torch.ones(1, 2, dtype=torch.bool)
    mask_b = torch.ones(1, 1, dtype=torch.bool)

    feats, _ = comp.compute_block(kin_a, kin_b, mask_a, mask_b, remove_self_pair=False)
    assert feats.shape == (1, 2, 1, 2)

    for row, (pt, eta, phi, m) in enumerate([(100.0, 0.0, 0.0, 10.0),
                                             (50.0, 1.0, 0.5, 5.0)]):
        pxa, pya, pza, ea = _four_vector(pt, eta, phi, m)
        pxb, pyb, pzb, eb = _four_vector(300.0, -1.0, 2.0, 80.0)
        dot = ea * eb - pxa * pxb - pya * pyb - pza * pzb
        # direct form
        assert abs(feats[0, row, 0, 1].item() - math.log(2 * dot)) < 1e-3
        # identity vs the m2 channel: 2 p_i.p_j = m2_ij - m_i^2 - m_j^2
        m2_ij = math.exp(feats[0, row, 0, 0].item())
        assert abs(feats[0, row, 0, 1].item()
                   - math.log(m2_ij - m ** 2 - 80.0 ** 2)) < 1e-2
    print("OK minkowski dot identity (direct + via m2 channel)")


def test_minkowski_symmetry_and_rectangular_shape():
    comp = PairwiseFeatureComputer(feature_names=("mink",))
    kin_a = _kin([100.0, 50.0], [0.0, 1.0], [0.0, 0.5], [10.0, 5.0])
    kin_b = _kin([300.0], [-1.0], [2.0], [80.0])
    mask_a = torch.ones(1, 2, dtype=torch.bool)
    mask_b = torch.ones(1, 1, dtype=torch.bool)

    feats, _ = comp.compute_block(kin_a, kin_b, mask_a, mask_b, remove_self_pair=False)
    assert feats.shape == (1, 2, 1, 1)
    feats_T, _ = comp.compute_block(kin_b, kin_a, mask_b, mask_a, remove_self_pair=False)
    assert torch.allclose(feats_T[0, 0, :, :], feats[0, :, 0, :], atol=1e-6)
    print("OK minkowski symmetry + rectangular shape")


def test_feature_set_parsing():
    assert parse_pairwise_feature_set("", 4) == ("kt", "z", "dr", "m2")
    assert parse_pairwise_feature_set("", 1) == ("kt",)
    assert parse_pairwise_feature_set("part4", 4) == ("kt", "z", "dr", "m2")
    assert parse_pairwise_feature_set("mink", 4) == ("mink",)
    assert parse_pairwise_feature_set("mink1", 4) == ("mink",)
    assert parse_pairwise_feature_set("part4+mink", 4) == ("kt", "z", "dr", "m2", "mink")
    assert parse_pairwise_feature_set("m2,mink", 4) == ("m2", "mink")
    assert parse_pairwise_feature_set("M2 , MINK", 4) == ("m2", "mink")  # case/space
    assert parse_pairwise_feature_set("m2+m2+mink", 4) == ("m2", "mink")  # dedup

    for bad_spec, bad_legacy in (("nope", 4), ("", 5), ("", 0), ("+", 4)):
        try:
            parse_pairwise_feature_set(bad_spec, bad_legacy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {(bad_spec, bad_legacy)}")

    assert set(CANONICAL_PAIRWISE_FEATURES) == {"kt", "z", "dr", "m2", "mink"}
    print("OK feature set parsing")


def test_feature_subset_matches_full():
    """Name-driven assembly must reproduce the legacy prefix cascade values."""
    torch.manual_seed(0)
    B, N = 2, 5
    pt = torch.rand(B, N) * 100
    eta = torch.randn(B, N)
    phi = torch.rand(B, N) * 2 * math.pi - math.pi
    mass = torch.rand(B, N) * 20
    mask = torch.ones(B, N, dtype=torch.bool)

    legacy = PairwiseFeatureComputer(num_features=4)
    named = PairwiseFeatureComputer(feature_names=("kt", "z", "dr", "m2"))
    f_legacy, _ = legacy(pt, eta, phi, mass, mask)
    f_named, _ = named(pt, eta, phi, mass, mask)
    assert torch.allclose(f_legacy, f_named, atol=1e-7)

    # A reordered subset picks the same values in its own order.
    sub = PairwiseFeatureComputer(feature_names=("m2", "kt"))
    f_sub, _ = sub(pt, eta, phi, mass, mask)
    assert torch.allclose(f_sub[..., 0], f_legacy[..., 3], atol=1e-7)
    assert torch.allclose(f_sub[..., 1], f_legacy[..., 0], atol=1e-7)
    print("OK subset matches legacy cascade")


def test_embedding_end_to_end_f5():
    torch.manual_seed(0)
    B, N, H = 2, 6, 4
    comp = PairwiseFeatureComputer(feature_names=("kt", "z", "dr", "m2", "mink"))
    assert comp.num_features == 5
    pt = torch.rand(B, N) * 100
    eta = torch.randn(B, N)
    phi = torch.rand(B, N) * 2 * math.pi - math.pi
    mass = torch.rand(B, N) * 20
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 4:] = False

    feats, pair_mask = comp(pt, eta, phi, mass, mask)
    assert feats.shape == (B, N, N, 5)

    emb = PairwiseEmbedding(num_features=5, num_heads=H, embed_dim=8).eval()
    bias = emb(feats, pair_mask)
    assert bias.shape == (B * H, N, N)
    bias = bias.view(B, H, N, N)
    assert torch.count_nonzero(bias[:, :, 4:, :]) == 0  # invalid rows zeroed
    assert torch.count_nonzero(bias[:, :, :, 4:]) == 0
    print("OK F=5 embedding end-to-end")


if __name__ == "__main__":
    test_minkowski_dot_identity()
    test_minkowski_symmetry_and_rectangular_shape()
    test_feature_set_parsing()
    test_feature_subset_matches_full()
    test_embedding_end_to_end_f5()
    print("\nAll pairwise Minkowski tests passed.")
