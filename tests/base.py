import os
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import numpy as np

# TODO: write description
class CtxGlue:
    def __init__(self, *ctxs):
        self.ctxs = ctxs

    def __enter__(self):
        for ctx in self.ctxs:
            ctx.__enter__()
    
    def __exit__(self, *args):
        for ctx in self.ctxs:
            ctx.__exit__(*args)

def _assert_equal(a: np.array, b: np.array):
    a, b = np.asanyarray(a).ravel(), np.asanyarray(b).ravel() #TODO: understand

    mismatches = np.where(a != b)[0]

    a_sample = a[mismatches[:5]]
    b_sample = b[mismatches[:5]]
    msg = f'''torch - c synth mismatch. {len(mismatches)} out of {len(a)} elements differ.
    Sample: {a_sample} vs {b_sample}'''
    assert len(mismatches) == 0, msg

class LayerTestBase:
    '''Base class for testing SPANet layers.'''

    custom_objects = {}

    @pytest.fixture
    def layer_kwargs(self, *args, **kwargs) -> dict:
        """Override this method to provide additional keyword arguments for the layer."""
        return {}

    @pytest.fixture
    def input_shapes(self, *args, **kwargs) -> Sequence[tuple]:
        """Override this method to provide input shapes for the layer."""
        raise NotImplementedError("input_shapes must be defined in the subclass")   
    
    @pytest.fixture
    def input_data(self, input_shapes, N: int = 1000)-> Sequence[torch.Tensor]:
        """Override this method to provide input data for the layer."""
        return [torch.randn(N, *shape) for shape in input_shapes]
    
    @pytest.fixture
    def model(self, layer, input_shapes):
        """Create test model with the given layer and input shapes."""
        if isinstance(input_shapes[0], int):
            input_shapes = (input_shapes,)
        inputs = [torch.randn(*shape) for shape in input_shapes]

        outputs = layer(*inputs)
        return torch.nn.Sequential(layer, outputs)
        