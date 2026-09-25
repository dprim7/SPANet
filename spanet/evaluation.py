import os
import re
import warnings
from glob import glob
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

from rich import progress

from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Evaluation, Outputs, Source
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions

from collections import defaultdict


def default_assignment_fn(outputs: Outputs):
    return extract_predictions([
        np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
        for assignment in outputs.assignments
    ])


def dict_concatenate(tree):
    output = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            output[key] = dict_concatenate(value)
        else:
            output[key] = np.concatenate(value)

    return output


def tree_concatenate(trees):
    leaves = []
    for tree in trees:
        data, tree_spec = tree_flatten(tree)
        leaves.append(data)

    results = [np.concatenate(l) for l in zip(*leaves)]
    return tree_unflatten(results, tree_spec)


def find_best_checkpoint(log_directory: str) -> str:
    """Return the path of the best checkpoint written by Lightning.

    The obvious `sorted(glob("checkpoints/epoch*"))[-1]` is wrong in two ways:
    filenames sort as TEXT, so "epoch=9" ranks above "epoch=17", and even when
    every retained epoch has the same number of digits it returns the newest of
    the retained checkpoints rather than the best-scoring one.

    Resolution order:

    1. Lightning stores its ModelCheckpoint state (including ``best_model_path``
       and the monitored metric) inside every checkpoint it writes. We read that
       state and re-base ``best_model_path`` onto ``log_directory`` by basename,
       so it still resolves when the run directory has been copied or moved off
       the training machine.
    2. Failing that, parse the monitored metric out of the filenames, which the
       default ``filename`` template embeds, and take the best according to the
       recorded mode.
    3. Failing that, fall back to the highest epoch/step ordered NUMERICALLY,
       and warn that this is the latest checkpoint rather than the best.
    """
    checkpoint_directory = os.path.join(log_directory, "checkpoints")
    candidates = glob(os.path.join(checkpoint_directory, "epoch*"))
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoint_directory}")

    # ---- 1. the checkpoint callback's own record of the best model ----------
    state_source = os.path.join(checkpoint_directory, "last.ckpt")
    if not os.path.exists(state_source):
        state_source = candidates[0]

    best_path, monitor, mode = None, None, None
    try:
        # weights_only defaults to True from torch 2.6, which refuses the plain
        # python objects in the callback state; ask for the full load explicitly
        # and stay compatible with versions that lack the argument.
        try:
            state = torch.load(state_source, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_source, map_location="cpu")
        for key, callback_state in (state.get("callbacks") or {}).items():
            if "ModelCheckpoint" not in str(key) or not isinstance(callback_state, dict):
                continue
            best_path = callback_state.get("best_model_path") or best_path
            monitor = callback_state.get("monitor") or monitor
            mode = callback_state.get("mode") or mode
            # the key itself carries the monitor/mode for most Lightning versions
            if monitor is None:
                found = re.search(r"'monitor': '([^']+)'", str(key))
                monitor = found.group(1) if found else None
            if mode is None:
                found = re.search(r"'mode': '([^']+)'", str(key))
                mode = found.group(1) if found else None
    except Exception as error:  # a corrupt or partial checkpoint must not be fatal
        warnings.warn(f"Could not read checkpoint callback state from {state_source}: {error}")

    if best_path:
        rebased = os.path.join(checkpoint_directory, os.path.basename(best_path))
        if os.path.exists(rebased):
            return rebased

    # ---- 2. parse the monitored metric out of the filenames -----------------
    if monitor:
        pattern = re.compile(re.escape(monitor) + r"=(-?[0-9]+\.?[0-9]*)")
        scored = []
        for path in candidates:
            found = pattern.search(os.path.basename(path))
            if found:
                scored.append((float(found.group(1)), path))
        if scored:
            pick = min if (mode or "max") == "min" else max
            return pick(scored, key=lambda item: item[0])[1]

    # ---- 3. numeric epoch/step ordering, clearly flagged --------------------
    def epoch_step(path: str):
        name = os.path.basename(path)
        epoch = re.search(r"epoch=(\d+)", name)
        step = re.search(r"step=(\d+)", name)
        return (int(epoch.group(1)) if epoch else -1,
                int(step.group(1)) if step else -1)

    latest = max(candidates, key=epoch_step)
    warnings.warn(
        "Could not determine the best checkpoint from the callback state or the "
        f"filenames; falling back to the LATEST checkpoint ({os.path.basename(latest)}). "
        "Pass an explicit checkpoint to avoid this."
    )
    return latest


def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    fp16: bool = False,
    checkpoint: Optional[str] = None,
    overrides: Optional[dict] = None
) -> JetReconstructionModel:
    # Load the best-performing checkpoint on validation data
    if checkpoint is None:
        checkpoint = find_best_checkpoint(log_directory)
        print(f"Loading: {checkpoint}")

    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint["state_dict"]
    if fp16:
        checkpoint = tree_map(lambda x: x.half(), checkpoint)

    # Load the options that were used for this run and set the testing-dataset value
    options = Options.load(f"{log_directory}/options.json")

    # Override options from command line arguments
    if testing_file is not None:
        options.testing_file = testing_file

    if event_info_file is not None:
        options.event_info_file = event_info_file

    if batch_size is not None:
        options.batch_size = batch_size

    if overrides is not None:
        for key, value in overrides.items():
            setattr(options, key, value)

    # Create model and disable all training operations for speed
    model = JetReconstructionModel(options)
    model.load_state_dict(checkpoint)
    model = model.eval().cpu().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def evaluate_on_test_dataset(
        model: JetReconstructionModel,
        progress=progress,
        return_full_output: bool = False,
        fp16: bool = False,
        assignment_fn=default_assignment_fn
) -> Union[Evaluation, Tuple[Evaluation, Outputs]]:
    full_assignments = defaultdict(list)
    full_assignment_probabilities = defaultdict(list)
    full_detection_probabilities = defaultdict(list)

    full_classifications = defaultdict(list)
    full_regressions = defaultdict(list)

    full_outputs = []

    dataloader = model.test_dataloader()
    if progress:
        dataloader = progress.track(model.test_dataloader(), description="Evaluating Model")

    for batch in dataloader:
        sources = tuple(Source(x[0].to(model.device), x[1].to(model.device)) for x in batch.sources)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs = model.forward(sources)

        assignment_indices = assignment_fn(outputs)

        detection_probabilities = np.stack([
            torch.sigmoid(detection).cpu().numpy()
            for detection in outputs.detections
        ])

        classifications = {
            key: torch.softmax(classification, 1).cpu().numpy()
            for key, classification in outputs.classifications.items()
        }

        regressions = {
            key: value.cpu().numpy()
            for key, value in outputs.regressions.items()
        }

        assignment_probabilities = []
        dummy_index = torch.arange(assignment_indices[0].shape[0])
        for assignment_probability, assignment, symmetries in zip(
            outputs.assignments,
            assignment_indices,
            model.event_info.product_symbolic_groups.values()
        ):
            # Get the probability of the best assignment.
            # Have to use explicit function call here to construct index dynamically.
            assignment_probability = assignment_probability.__getitem__((dummy_index, *assignment.T))

            # Convert from log-probability to probability.
            assignment_probability = torch.exp(assignment_probability)

            # Multiply by the symmetry factor to account for equivalent predictions.
            assignment_probability = symmetries.order() * assignment_probability

            # Convert back to cpu and add to database.
            assignment_probabilities.append(assignment_probability.cpu().numpy())

        for i, name in enumerate(model.event_info.product_particles):
            full_assignments[name].append(assignment_indices[i])
            full_assignment_probabilities[name].append(assignment_probabilities[i])
            full_detection_probabilities[name].append(detection_probabilities[i])

        for key, regression in regressions.items():
            full_regressions[key].append(regression)

        for key, classification in classifications.items():
            full_classifications[key].append(classification)

        if return_full_output:
            full_outputs.append(tree_map(lambda x: x.cpu().numpy(), outputs))

    evaluation = Evaluation(
        dict_concatenate(full_assignments),
        dict_concatenate(full_assignment_probabilities),
        dict_concatenate(full_detection_probabilities),
        dict_concatenate(full_regressions),
        dict_concatenate(full_classifications)
    )

    if return_full_output:
        return evaluation, tree_concatenate(full_outputs)

    return evaluation
