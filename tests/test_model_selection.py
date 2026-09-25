"""Regression tests for best-checkpoint selection and weighted accuracy dtype.

Both behaviours are exercised with real filenames taken from production runs, so
a regression reproduces the concrete mis-selection observed in practice rather
than an invented corner case.
"""
import os

import numpy as np
import pytest
import torch

from spanet.evaluation import find_best_checkpoint

# real filenames from a sweep trial: the best epoch (24) is NOT the string-max,
# because "epoch=9" sorts above "epoch=24"
REAL_CASE = [
    "epoch=9-step=16030-validation_average_jet_accuracy=0.387.ckpt",
    "epoch=15-step=25648-validation_average_jet_accuracy=0.421.ckpt",
    "epoch=24-step=40075-validation_average_jet_accuracy=0.429.ckpt",
]
BEST = "epoch=24-step=40075-validation_average_jet_accuracy=0.429.ckpt"
MONITOR = "validation_average_jet_accuracy"


def _make_run(tmp_path, names, callback_state=None, last=False):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": {}}
    if callback_state is not None:
        payload["callbacks"] = callback_state
    for name in names:
        torch.save({"state_dict": {}}, ckpt_dir / name)
    if last:
        torch.save(payload, ckpt_dir / "last.ckpt")
    else:  # fold the state into the epoch checkpoints instead
        for name in names:
            torch.save(payload, ckpt_dir / name)
    return str(tmp_path)


def test_string_sort_would_pick_the_wrong_file():
    """Guards the premise: the old implementation really does mis-sort."""
    assert sorted(REAL_CASE)[-1] != BEST
    assert sorted(REAL_CASE)[-1].startswith("epoch=9")


def test_uses_callback_best_model_path(tmp_path):
    state = {"ModelCheckpoint{'monitor': 'x', 'mode': 'max'}":
             {"best_model_path": f"/some/other/machine/checkpoints/{BEST}",
              "monitor": MONITOR, "mode": "max"}}
    run = _make_run(tmp_path, REAL_CASE, state, last=True)
    assert os.path.basename(find_best_checkpoint(run)) == BEST


def test_rebases_best_path_when_run_directory_moved(tmp_path):
    """best_model_path is absolute on the training machine; files get rsynced."""
    state = {"ModelCheckpoint": {"best_model_path": f"/scratch/logs/run/checkpoints/{BEST}",
                                 "monitor": MONITOR, "mode": "max"}}
    run = _make_run(tmp_path, REAL_CASE, state, last=True)
    assert os.path.basename(find_best_checkpoint(run)) == BEST


def test_falls_back_to_parsing_the_metric_from_filenames(tmp_path):
    # callback state present but best_model_path missing/stale
    state = {"ModelCheckpoint": {"best_model_path": "/gone/nonexistent.ckpt",
                                 "monitor": MONITOR, "mode": "max"}}
    run = _make_run(tmp_path, REAL_CASE, state, last=True)
    assert os.path.basename(find_best_checkpoint(run)) == BEST


def test_min_mode_is_respected(tmp_path):
    names = ["epoch=9-step=1-val_loss=0.90.ckpt",
             "epoch=24-step=2-val_loss=0.10.ckpt"]
    state = {"ModelCheckpoint": {"best_model_path": "", "monitor": "val_loss", "mode": "min"}}
    run = _make_run(tmp_path, names, state, last=True)
    assert "val_loss=0.10" in os.path.basename(find_best_checkpoint(run))


def test_last_resort_is_numeric_not_lexicographic(tmp_path):
    names = ["epoch=9-step=16030.ckpt", "epoch=24-step=40075.ckpt"]
    run = _make_run(tmp_path, names, callback_state=None, last=False)
    with pytest.warns(UserWarning, match="LATEST checkpoint"):
        picked = find_best_checkpoint(run)
    assert "epoch=24" in os.path.basename(picked)   # numeric, not "epoch=9"


def test_missing_checkpoints_raises(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint(str(tmp_path))


def test_weighted_accuracy_dtype_preserves_weights():
    """The metric array stores accuracy*weight; a bool dtype truncates it."""
    from spanet.network.jet_reconstruction import jet_reconstruction_validation as v
    import inspect
    src = inspect.getsource(v.JetReconstructionValidation.compute_metrics)
    assert "weighted_jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.float64)" in src, \
        "weighted_jet_accuracies must be float; bool silently truncates non-unit weights"

    # and demonstrate why
    correct = np.array([[True, True, False]])
    weight = np.array([[2.5, 0.2, 1.0]])
    as_bool = np.zeros((1, 3), dtype=bool);    as_bool[0] = correct[0] * weight[0]
    as_float = np.zeros((1, 3), dtype=np.float64); as_float[0] = correct[0] * weight[0]
    assert as_bool.sum() == 2.0            # weights destroyed
    assert as_float.sum() == pytest.approx(2.7)
