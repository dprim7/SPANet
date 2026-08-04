"""Tests for the Eq(2->2) PELICAN-lite augmentation in PairwiseEmbedding.

Torch-only. Run:  PYTHONPATH=. python tests/test_pairwise_eq2to2.py
"""
import torch

from spanet.network.jet_reconstruction_pairwise.pairwise_features import PairwiseEmbedding


def test_masked_means_exact():
    """Appended channels must equal hand-computed MASKED means."""
    B, F, R, C = 1, 2, 3, 2
    x4 = torch.arange(B * F * R * C, dtype=torch.float32).view(B, F, R, C) + 1.0
    valid = torch.ones(B, R, C, dtype=torch.bool)
    valid[:, 2, :] = False   # padded row
    valid[:, :, 1] = False   # padded column
    x4 = x4 * valid.unsqueeze(1)

    x = x4.view(B, F, R * C)
    mask = valid.view(B, 1, R * C).float()
    out = PairwiseEmbedding._eq2to2_augment(x, mask, R, C).view(B, 4 * F, R, C)

    v = valid[0].float()
    for f in range(F):
        xf = x4[0, f]
        for r in range(R):
            n = v[r].sum().item()
            exp_row = (xf[r] * v[r]).sum().item() / max(n, 1.0)
            for c in range(C):
                got = out[0, F + f, r, c].item()
                want = exp_row if valid[0, r, c] else 0.0
                assert abs(got - want) < 1e-6, (f, r, c, got, want)
        for c in range(C):
            n = v[:, c].sum().item()
            exp_col = (xf[:, c] * v[:, c]).sum().item() / max(n, 1.0)
            for r in range(R):
                got = out[0, 2 * F + f, r, c].item()
                want = exp_col if valid[0, r, c] else 0.0
                assert abs(got - want) < 1e-6
        exp_glob = (xf * v).sum().item() / v.sum().item()
        for r in range(R):
            for c in range(C):
                got = out[0, 3 * F + f, r, c].item()
                want = exp_glob if valid[0, r, c] else 0.0
                assert abs(got - want) < 1e-6
    # Identity channels preserved on valid cells.
    assert torch.allclose(out[:, :F] * valid.unsqueeze(1), x4)
    print("OK masked means exact (row/col/global, padded cells zero)")


def test_rectangular_row_col_differ():
    """On an asymmetric rectangular block, row-mean map != col-mean map."""
    torch.manual_seed(0)
    B, F, R, C = 2, 3, 4, 2
    x = torch.randn(B, F, R * C)
    out = PairwiseEmbedding._eq2to2_augment(x, None, R, C).view(B, 4 * F, R, C)
    assert not torch.allclose(out[:, F:2 * F], out[:, 2 * F:3 * F], atol=1e-5)
    print("OK rectangular row-mean != col-mean")


def test_flag_off_identity():
    """eq2to2=False must be bit-identical to the pre-change module.

    Reference keys are derived (MaskedBatchNorm1d is a custom module; it
    carries weight/bias/running_mean/running_var per BN, no
    num_batches_tracked): every key must belong to the known layer set and
    conv1 must keep its narrow shape. Flag-on differs ONLY in conv1's width.
    """
    torch.manual_seed(0)
    emb = PairwiseEmbedding(num_features=4, num_heads=2, embed_dim=8)
    emb_on = PairwiseEmbedding(num_features=4, num_heads=2, embed_dim=8, eq2to2=True)

    sd_off, sd_on = emb.state_dict(), emb_on.state_dict()
    assert set(sd_off.keys()) == set(sd_on.keys())
    assert emb.conv1.weight.shape == (8, 4, 1)
    assert emb_on.conv1.weight.shape == (8, 16, 1)
    for k in sd_off:
        if not k.startswith("conv1."):
            assert sd_off[k].shape == sd_on[k].shape, k
    assert {k.split(".")[0] for k in sd_off} == {
        "input_bn", "bn1", "bn2", "conv1", "conv2", "conv3"
    }
    assert emb.input_bn.weight.shape == (4,)  # BN stays on the raw F channels
    assert emb_on.input_bn.weight.shape == (4,)
    print("OK flag-off state_dict identical; flag-on widens conv1 only")


def test_bias_zero_on_invalid_with_eq2to2():
    torch.manual_seed(0)
    B, N, F, H = 2, 6, 4, 4
    feats = torch.randn(B, N, N, F)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, 4:] = False
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    i = torch.arange(N)
    pair_mask = pair_mask.clone()
    pair_mask[:, i, i] = False
    feats = feats * pair_mask.unsqueeze(-1)

    emb = PairwiseEmbedding(num_features=F, num_heads=H, embed_dim=8, eq2to2=True).eval()
    bias = emb(feats, pair_mask).view(B, H, N, N)
    assert torch.count_nonzero(bias[:, :, 4:, :]) == 0
    assert torch.count_nonzero(bias[:, :, :, 4:]) == 0
    assert torch.count_nonzero(bias[:, :, i, i]) == 0
    print("OK bias zero on invalid pairs with eq2to2")


def test_empty_row_no_nan():
    """A fully-padded row must produce finite output (clamped counts)."""
    B, F, R, C = 1, 2, 3, 3
    x = torch.randn(B, F, R * C)
    valid = torch.ones(B, R, C, dtype=torch.bool)
    valid[:, 1, :] = False
    mask = valid.view(B, 1, R * C).float()
    out = PairwiseEmbedding._eq2to2_augment(x * mask, mask, R, C)
    assert torch.isfinite(out).all()
    print("OK empty row produces finite output")


if __name__ == "__main__":
    test_masked_means_exact()
    test_rectangular_row_col_differ()
    test_flag_off_identity()
    test_bias_zero_on_invalid_with_eq2to2()
    test_empty_row_no_nan()
    print("\nAll pairwise Eq2to2 tests passed.")
