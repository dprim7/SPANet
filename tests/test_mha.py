"""
Unit tests for SPANet Symmetric Multi-Head Attention layers.

This module tests the core symmetric attention systems:
- SymmetricAttentionFull: Full tensor contraction approach
- SymmetricAttentionSplit: Split attention with encoders

Tests verify:
1. Layer initialization and parameter setup
2. Forward pass with different input configurations
3. Output shape consistency
4. Gradient flow
5. Symmetry preservation

Run with: pytest tests/test_mha.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from spanet.dataset.types import Source, SpecialKey
from spanet.network.symmetric_attention import SymmetricAttentionFull, SymmetricAttentionSplit
from spanet.options import Options

from tests.base import LayerTestBase


class TestSymmetricAttentionFull(LayerTestBase):
    """Test the SymmetricAttentionFull layer."""

    layer_cls = SymmetricAttentionFull

    @pytest.fixture(params=[2])  # Test degree-2 (pairwise) attention
    def degree(self, request):
        return request.param

    @pytest.fixture
    def options(self):
        """Create minimal Options for testing."""
        options = Options()
        options.hidden_dim = 64
        options.batch_size = 4
        options.num_attention_heads = 1  # Simplified for testing
        return options

    @pytest.fixture
    def layer_kwargs(self, options, degree):
        """Provide constructor arguments for SymmetricAttentionFull."""
        return {
            'options': options,
            'degree': degree,
            'permutation_indices': None,  # No custom symmetries for basic test
            'attention_dim': None  # Use default (hidden_dim)
        }

    @pytest.fixture
    def input_shapes(self, options):
        """Define input shapes for attention layer.
        
        SymmetricAttention expects:
        - x: [T, B, D] (sequence_length, batch_size, hidden_dim)
        - padding_mask: [B, T] (batch_size, sequence_length)
        - sequence_mask: [T, B, 1] (sequence_length, batch_size, 1)
        """
        seq_len = 6
        batch_size = options.batch_size
        hidden_dim = options.hidden_dim
        
        return {
            'x_shape': (seq_len, batch_size, hidden_dim),
            'padding_mask_shape': (batch_size, seq_len),
            'sequence_mask_shape': (seq_len, batch_size, 1),
            'seq_len': seq_len,
            'batch_size': batch_size,
            'hidden_dim': hidden_dim
        }

    @pytest.fixture
    def layer(self, layer_kwargs):
        """Create the attention layer."""
        return self.layer_cls(**layer_kwargs)

    @pytest.fixture
    def sample_inputs(self, input_shapes):
        """Create sample input tensors."""
        shapes = input_shapes
        
        # Input features: [T, B, D]
        x = torch.randn(shapes['x_shape'], dtype=torch.float32)
        
        # Padding mask: [B, T] - False means padding
        padding_mask = torch.zeros(shapes['padding_mask_shape'], dtype=torch.bool)
        # Make first few positions real (not padding)
        padding_mask[:, :4] = False  # First 4 positions are real
        padding_mask[:, 4:] = True   # Rest are padding
        
        # Sequence mask: [T, B, 1] - True means real
        sequence_mask = torch.ones(shapes['sequence_mask_shape'], dtype=torch.bool)
        sequence_mask[4:, :, :] = False  # Last positions are padding
        
        return {
            'x': x,
            'padding_mask': padding_mask,
            'sequence_mask': sequence_mask
        }

    def test_layer_initialization(self, layer, options, degree):
        """Test that layer initializes correctly."""
        assert isinstance(layer, (SymmetricAttentionFull, SymmetricAttentionSplit))
        assert layer.degree == degree
        assert layer.features == options.hidden_dim
        assert layer.batch_size == options.batch_size
        
        # Check weights are initialized (Full has weights parameter, Split has linear layers)
        if isinstance(layer, SymmetricAttentionFull):
            assert hasattr(layer, 'weights')
            expected_weight_shape = [options.hidden_dim] * degree
            assert list(layer.weights.shape) == expected_weight_shape
        else:  # SymmetricAttentionSplit
            assert hasattr(layer, 'linear_layers')
            assert len(layer.linear_layers) == degree

    def test_forward_pass_basic(self, layer, sample_inputs, input_shapes, degree):
        """Test basic forward pass."""
        inputs = sample_inputs
        
        # Forward pass
        output = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask']
        )
        
        # Check output shape
        batch_size = input_shapes['batch_size']
        seq_len = input_shapes['seq_len']
        expected_shape = [batch_size] + [seq_len] * degree
        
        assert list(output.shape) == expected_shape
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

    def test_forward_pass_with_attention_bias(self, layer, sample_inputs, input_shapes, degree):
        """Test forward pass with attention bias."""
        inputs = sample_inputs
        batch_size = input_shapes['batch_size']
        seq_len = input_shapes['seq_len']
        
        # Create attention bias
        attention_bias = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Forward pass with bias
        output_with_bias = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask'],
            attention_bias=attention_bias
        )
        
        # Forward pass without bias
        output_no_bias = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask']
        )
        
        # Handle different return types (Full returns tensor, Split returns tuple)
        if isinstance(output_with_bias, tuple):
            output_with_bias_tensor = output_with_bias[0]
            output_no_bias_tensor = output_no_bias[0]
        else:
            output_with_bias_tensor = output_with_bias
            output_no_bias_tensor = output_no_bias
        
        # Outputs should be different
        assert not torch.allclose(output_with_bias_tensor, output_no_bias_tensor)
        
        # Shapes should be the same
        assert output_with_bias_tensor.shape == output_no_bias_tensor.shape

    def test_gradient_flow(self, layer, sample_inputs):
        """Test that gradients flow correctly."""
        inputs = sample_inputs

        # Ensure inputs require gradients
        x = inputs['x'].clone().requires_grad_(True)

        # Forward pass
        output = layer(x, inputs['padding_mask'], inputs['sequence_mask'])
        
        # Handle different return types (Full returns tensor, Split returns tuple)
        if isinstance(output, tuple):
            output_tensor = output[0]  # Get the main output tensor
        else:
            output_tensor = output

        # Backward pass
        loss = output_tensor.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_symmetry_preservation(self, layer, sample_inputs, degree):
        """Test that output respects permutation symmetries."""
        if degree != 2:
            pytest.skip("Symmetry test only implemented for degree=2")

        inputs = sample_inputs

        # Get original output
        output = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask']
        )
        
        # Handle different return types (Full returns tensor, Split returns tuple)
        if isinstance(output, tuple):
            output_tensor = output[0]  # Get the main output tensor
        else:
            output_tensor = output

        # Apply permutation to inputs (swap jets 0 and 1)
        perm_indices = torch.tensor([1, 0, 2, 3, 4, 5])  # Swap first two jets

        x_perm = inputs['x'][perm_indices]
        padding_mask_perm = inputs['padding_mask'][:, perm_indices]
        sequence_mask_perm = inputs['sequence_mask'][perm_indices]

        # Get permuted output
        output_perm = layer(x_perm, padding_mask_perm, sequence_mask_perm)
        
        # Handle different return types for permuted output
        if isinstance(output_perm, tuple):
            output_perm_tensor = output_perm[0]
        else:
            output_perm_tensor = output_perm

        # Apply same permutation to output dimensions
        output_expected = output_tensor[:, perm_indices][:, :, perm_indices]
        
        # Should be approximately equal (within numerical precision)
        assert torch.allclose(output_perm_tensor, output_expected, atol=1e-5)


class TestSymmetricAttentionSplit(TestSymmetricAttentionFull):
    """Test the SymmetricAttentionSplit layer."""

    layer_cls = SymmetricAttentionSplit

    @pytest.fixture
    def options(self):
        """Create Options with additional parameters for Split attention."""
        options = Options()
        options.hidden_dim = 64
        options.batch_size = 4
        options.num_attention_heads = 1
        # Split attention specific options
        options.num_jet_embedding_layers = 2
        options.num_jet_encoder_layers = 1
        options.masking = "Filling"  # Required for masking layer
        return options

    def test_forward_pass_basic(self, layer, sample_inputs, input_shapes, degree):
        """Test basic forward pass for Split attention."""
        inputs = sample_inputs
        
        # Forward pass - Split attention returns (output, daughter_vectors)
        result = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask']
        )
        
        # Unpack results
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        output, daughter_vectors = result
        
        # Check output shape
        batch_size = input_shapes['batch_size']
        seq_len = input_shapes['seq_len']
        expected_shape = [batch_size] + [seq_len] * degree
        
        assert list(output.shape) == expected_shape
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        
        # Check daughter vectors
        assert isinstance(daughter_vectors, list)
        assert len(daughter_vectors) == degree

    def test_forward_pass_with_attention_bias(self, layer, sample_inputs, input_shapes, degree):
        """Test forward pass with attention bias for Split attention."""
        inputs = sample_inputs
        batch_size = input_shapes['batch_size']
        seq_len = input_shapes['seq_len']
        
        # Create attention bias
        attention_bias = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Forward pass with bias
        output_with_bias, _ = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask'],
            attention_bias=attention_bias
        )
        
        # Forward pass without bias
        output_no_bias, _ = layer(
            inputs['x'],
            inputs['padding_mask'],
            inputs['sequence_mask']
        )
        
        # Outputs should be different
        assert not torch.allclose(output_with_bias, output_no_bias)
        
        # Shapes should be the same
        assert output_with_bias.shape == output_no_bias.shape


class TestAttentionLayerComparison:
    """Compare behavior between Full and Split attention."""
    
    @pytest.fixture
    def options(self):
        """Shared options for comparison."""
        options = Options()
        options.hidden_dim = 32  # Smaller for faster testing
        options.batch_size = 2
        options.num_jet_embedding_layers = 1
        options.num_jet_encoder_layers = 1
        options.masking = "Filling"
        return options

    @pytest.fixture
    def common_inputs(self, options):
        """Common inputs for both layers."""
        batch_size = options.batch_size
        seq_len = 4
        hidden_dim = options.hidden_dim
        
        x = torch.randn(seq_len, batch_size, hidden_dim)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        
        return {
            'x': x,
            'padding_mask': padding_mask,
            'sequence_mask': sequence_mask
        }

    def test_output_shape_consistency(self, options, common_inputs):
        """Test that both layers produce outputs with the same shape."""
        degree = 2
        
        # Create both layers
        full_layer = SymmetricAttentionFull(
            options=options,
            degree=degree,
            permutation_indices=None
        )
        
        split_layer = SymmetricAttentionSplit(
            options=options,
            degree=degree,
            permutation_indices=None
        )
        
        # Forward passes
        full_output = full_layer(**common_inputs)
        split_output, _ = split_layer(**common_inputs)
        
        # Shapes should match
        assert full_output.shape == split_output.shape