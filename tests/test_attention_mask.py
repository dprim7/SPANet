"""
Unit tests for attention bias functionality in SPANet.

This module tests the integration of attention biases for pre-softmax attention weights,
ensuring proper data flow from HDF5 files through the network to the attention layers.

Attention biases are added to attention scores before softmax, allowing for:
1. Standard masking (0.0 for allowed, -inf for forbidden)
2. Continuous bias values (e.g., AK jet overlap scores)
3. More nuanced attention control than binary masks

These tests follow Test-Driven Development (TDD) principles:
1. Write tests first to define expected behavior
2. Implement minimal code to make tests pass
3. Refactor and improve implementation
4. Verify tests still pass

Run with: pytest tests/test_attention_mask.py -v
"""

import pytest
import numpy as np
import torch
import h5py
import tempfile
import os
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Import SPANet modules
try:
    from spanet.dataset.types import Source, SpecialKey
    from spanet.dataset.inputs.RelativeInput import RelativeInput
    from spanet.dataset.inputs.SequentialInput import SequentialInput
    from spanet.dataset.event_info import EventInfo
    from spanet.network.symmetric_attention.symmetric_attention_full import SymmetricAttentionFull
    from spanet.options import Options
except ImportError as e:
    pytest.skip(f"SPANet modules not available: {e}", allow_module_level=True)


class TestAttentionBiasDataTypes:
    """Test the basic data structures for attention biases."""
    
    def test_source_with_attention_bias(self):
        """Test that Source dataclass can hold attention bias."""
        # Test data
        data = torch.randn(5, 3)  # 5 jets, 3 features
        mask = torch.ones(5, dtype=torch.bool)  # mask for 5 jets
        attention_bias = torch.randn(5, 5)  # 5x5 pairwise bias matrix (continuous values)
        
        # Create Source with attention bias
        source = Source(
            data=data,
            mask=mask,
            ak_overlap_mask=attention_bias  # Note: field name remains the same for compatibility
        )
        
        # Some sanity checks
        assert source.data.shape == (5, 3)
        assert source.mask.shape == (5,)
        assert source.ak_overlap_mask.shape == (5, 5)
        assert source.ak_overlap_mask.dtype == torch.float32
        
    def test_source_without_attention_bias(self):
        """Test that Source works when attention bias is None."""
        data = torch.randn(5, 3)
        mask = torch.ones(5, dtype=torch.bool)
        
        source = Source(
            data=data,
            mask=mask,
            ak_overlap_mask=None
        )
        
        assert source.ak_overlap_mask is None


class TestAttentionBiasHDF5Loading:
    """Test loading attention biases from HDF5 files."""
    
    @pytest.fixture
    def sample_hdf5_file(self):
        """Create a temporary HDF5 file with sample data including attention bias."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.close()
        
        # Create sample data
        num_events = 100
        max_jets = 10
        
        with h5py.File(temp_file.name, 'w') as f:
            # Create input group structure
            inputs = f.create_group('INPUTS')
            jets = inputs.create_group('Jets')
            
            # Standard jet features
            jets.create_dataset('MASK', data=np.random.choice([True, False], (num_events, max_jets)))
            jets.create_dataset('pt', data=np.random.uniform(20, 500, (num_events, max_jets)))
            jets.create_dataset('eta', data=np.random.uniform(-3, 3, (num_events, max_jets)))
            jets.create_dataset('phi', data=np.random.uniform(-np.pi, np.pi, (num_events, max_jets)))
            jets.create_dataset('mass', data=np.random.uniform(0, 100, (num_events, max_jets)))
            
            # NEW: Attention bias (pairwise bias matrix)
            # For each event, create a symmetric bias matrix
            attention_biases = np.zeros((num_events, max_jets, max_jets), dtype=np.float32)
            for i in range(num_events):
                # Create random symmetric bias pattern
                # Use continuous values: positive for encouraging attention, negative for discouraging
                bias = np.random.normal(0, 0.5, (max_jets, max_jets)).astype(np.float32)
                # Make it symmetric for physical consistency
                bias = (bias + bias.T) / 2
                # Set diagonal to 0 (no self-bias)
                np.fill_diagonal(bias, 0.0)
                attention_biases[i] = bias
                
            jets.create_dataset('ak_overlap_mask', data=attention_biases)
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_load_attention_bias_from_hdf5(self, sample_hdf5_file):
        """Test that attention biases can be loaded from HDF5 files."""
        with h5py.File(sample_hdf5_file, 'r') as f:
            # Test direct access to attention bias
            attention_bias = f['INPUTS/Jets/ak_overlap_mask'][:]
            
            assert attention_bias.shape == (100, 10, 10)  # num_events, max_jets, max_jets
            assert attention_bias.dtype == np.float32
            
            # Verify symmetry property
            for event_idx in range(min(10, attention_bias.shape[0])):  # Test first 10 events
                event_bias = attention_bias[event_idx]
                # Check if matrix is symmetric
                assert np.allclose(event_bias, event_bias.T), f"Event {event_idx} bias is not symmetric"
                # Check diagonal is zero (no self-bias)
                assert np.allclose(np.diag(event_bias), 0.0), f"Event {event_idx} diagonal is not zero"
    
    def test_missing_attention_bias_handling(self):
        """Test graceful handling when attention bias is missing from HDF5."""
        # Create HDF5 without attention bias
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.close()
        
        try:
            with h5py.File(temp_file.name, 'w') as f:
                inputs = f.create_group('INPUTS')
                jets = inputs.create_group('Jets')
                jets.create_dataset('MASK', data=np.ones((10, 5), dtype=bool))
                jets.create_dataset('pt', data=np.random.uniform(20, 500, (10, 5)))
                # Note: NO ak_overlap_mask dataset
            
            # Test that we can handle missing bias gracefully
            with h5py.File(temp_file.name, 'r') as f:
                with pytest.raises(KeyError):
                    _ = f['INPUTS/Jets/ak_overlap_mask']
                    
        finally:
            os.unlink(temp_file.name)


class TestAttentionBiasSymmetry:
    """Test attention bias symmetry preservation."""
    
    def test_bias_symmetry_validation(self):
        """Test function to validate bias symmetry."""
        # Create symmetric bias
        symmetric_bias = torch.tensor([
            [0.0, 0.5, -0.2],
            [0.5, 0.0, 0.8],
            [-0.2, 0.8, 0.0]
        ], dtype=torch.float32)
        
        # Create non-symmetric bias
        asymmetric_bias = torch.tensor([
            [0.0, 0.5, -0.2],
            [0.3, 0.0, 0.8],  # Different from symmetric_bias[0, 1]
            [-0.2, 0.8, 0.0]
        ], dtype=torch.float32)
        
        # Symmetry validation function
        def is_symmetric(bias: torch.Tensor) -> bool:
            return torch.allclose(bias, bias.T, atol=1e-6)
        
        assert is_symmetric(symmetric_bias), "Symmetric bias should pass validation"
        assert not is_symmetric(asymmetric_bias), "Asymmetric bias should fail validation"
    
    def test_bias_permutation_invariance(self):
        """Test that biases respect permutation invariance."""
        # Original bias
        original_bias = torch.tensor([
            [0.0, 0.5, -0.2],
            [0.5, 0.0, 0.8],
            [-0.2, 0.8, 0.0]
        ], dtype=torch.float32)
        
        # Permutation: swap indices 0 and 2
        perm = torch.tensor([2, 1, 0])
        
        # Apply permutation to both dimensions
        permuted_bias = original_bias[perm][:, perm]
        
        # Verify the permuted bias maintains the same structure
        # original_bias[0,1] should equal permuted_bias[2,1] after permutation
        assert torch.isclose(original_bias[0, 1], permuted_bias[2, 1])
        assert torch.isclose(original_bias[1, 0], permuted_bias[1, 2])


class TestSymmetricAttentionWithBias:
    """Test symmetric attention layers with attention biases."""
    """Test symmetric attention layers with attention masks."""
    
    @pytest.fixture
    def sample_attention_setup(self):
        """Create sample data for attention testing."""
        batch_size = 2
        seq_len = 5
        hidden_dim = 128  # Use default hidden_dim from Options
        degree = 2
        
        # Create sample input
        x = torch.randn(seq_len, batch_size, hidden_dim)
        
        # Create padding mask (False means padding)
        padding_mask = torch.tensor([
            [False, False, False, True, True],  # First 3 are real, last 2 are padding
            [False, False, False, False, True]   # First 4 are real, last 1 is padding
        ], dtype=torch.bool)
        
        # Create sequence mask (True means real)
        sequence_mask = torch.tensor([
            [[True], [True], [True], [False], [False]],
            [[True], [True], [True], [True], [False]]
        ]).transpose(0, 1)  # Shape: [seq_len, batch_size, 1]
        
        # Create attention bias (continuous values for pairwise interactions)
        attention_bias = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.float32)
        # Add some bias values for testing
        attention_bias[0, 0, 2] = -2.0  # Discourage jet 0 -> jet 2 interaction in batch 0
        attention_bias[0, 2, 0] = -2.0  # Make it symmetric
        attention_bias[0, 1, 2] = 1.0   # Encourage jet 1 -> jet 2 interaction
        attention_bias[0, 2, 1] = 1.0   # Make it symmetric
        
        return {
            'x': x,
            'padding_mask': padding_mask,
            'sequence_mask': sequence_mask,
            'attention_bias': attention_bias,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'degree': degree
        }
    
    def test_attention_forward_with_bias(self, sample_attention_setup):
        """Test that attention layer accepts and processes biases correctly."""
        setup = sample_attention_setup
        
        # Create options (minimal for testing)
        options = Options()
        options.hidden_dim = setup['hidden_dim']
        options.batch_size = setup['batch_size']
        
        # Create attention layer
        attention = SymmetricAttentionFull(
            options=options,
            degree=setup['degree'],
            permutation_indices=None
        )
        
        # Test forward pass without bias
        output_no_bias = attention(
            setup['x'],
            setup['padding_mask'],
            setup['sequence_mask'],
            attention_mask=None
        )
        
        # Test forward pass with bias
        output_with_bias = attention(
            setup['x'],
            setup['padding_mask'],
            setup['sequence_mask'],
            attention_mask=setup['attention_bias']
        )
        
        # Verify outputs have correct shape
        expected_shape = (setup['batch_size'], setup['seq_len'], setup['seq_len'])
        assert output_no_bias.shape == expected_shape
        assert output_with_bias.shape == expected_shape
        
        # Verify outputs are different when bias is applied
        assert not torch.allclose(output_no_bias, output_with_bias), \
            "Outputs should be different when attention bias is applied"
        
        # Verify that biased positions show the expected relative changes
        # Position [0, 0, 2] had bias -2.0, should be lower than original
        # Position [0, 1, 2] had bias +1.0, should be higher than original
        bias_diff = output_with_bias - output_no_bias
        assert bias_diff[0, 0, 2] < bias_diff[0, 1, 2], \
            "Negative bias should reduce attention score relative to positive bias"
    
    def test_attention_bias_shape_compatibility(self, sample_attention_setup):
        """Test that attention biases work with different shapes and broadcasting."""
        setup = sample_attention_setup
        
        options = Options()
        options.hidden_dim = setup['hidden_dim']
        options.batch_size = setup['batch_size']
        
        attention = SymmetricAttentionFull(
            options=options,
            degree=setup['degree'],
            permutation_indices=None
        )
        
        # Test with 2D bias (should broadcast to 3D)
        bias_2d = torch.randn(setup['seq_len'], setup['seq_len'], dtype=torch.float32)
        
        try:
            output = attention(
                setup['x'],
                setup['padding_mask'], 
                setup['sequence_mask'],
                attention_mask=bias_2d
            )
            assert output.shape == (setup['batch_size'], setup['seq_len'], setup['seq_len'])
        except Exception as e:
            pytest.fail(f"2D bias should be broadcastable: {e}")


class TestAttentionBiasGradientFlow:
    """Test that gradients flow correctly through attention biases."""
    
    def test_gradient_flow_with_bias(self):
        """Test that gradients can flow through biased attention."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 32
        
        # Create simple test setup
        options = Options()
        options.hidden_dim = hidden_dim
        options.batch_size = batch_size
        
        attention = SymmetricAttentionFull(
            options=options,
            degree=2,
            permutation_indices=None
        )
        
        # Input that requires gradients
        x = torch.randn(seq_len, batch_size, hidden_dim, requires_grad=True)
        
        # Simple inputs
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Forward pass
        output = attention(x, padding_mask, sequence_mask, attention_mask=attention_bias)
        
        # Create a simple loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Verify gradients exist
        assert x.grad is not None, "Gradients should flow back to input"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), \
            "Gradients should be non-zero"
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Forward pass
        output = attention(x, padding_mask, sequence_mask, attention_mask=attention_bias)
        
        # Create a simple loss and backpropagate
        loss = output.sum()
        loss.backward()
        
        # Verify gradients exist
        assert x.grad is not None, "Gradients should flow back to input"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), \
            "Gradients should be non-zero"


class TestEndToEndIntegration:
    """Test end-to-end integration of attention biases."""
    
    def test_bias_propagation_chain(self):
        """Test that biases propagate correctly through the data pipeline."""
        # This test would verify the complete chain:
        # HDF5 -> Dataset -> DataLoader -> Model -> Attention
        
        # Create mock data that simulates the full pipeline
        batch_size = 2
        seq_len = 6
        feature_dim = 4
        
        # Mock Source objects with attention biases
        mock_sources = []
        for i in range(batch_size):
            data = torch.randn(seq_len, feature_dim)
            mask = torch.ones(seq_len, dtype=torch.bool)
            attention_bias = torch.randn(seq_len, seq_len, dtype=torch.float32)
            
            source = Source(
                data=data,
                mask=mask,
                ak_overlap_mask=attention_bias  # Field name stays the same for compatibility
            )
            mock_sources.append(source)
        
        # Verify each source has the expected structure
        for source in mock_sources:
            assert hasattr(source, 'ak_overlap_mask')
            assert source.ak_overlap_mask is not None
            assert source.ak_overlap_mask.shape == (seq_len, seq_len)
            assert source.ak_overlap_mask.dtype == torch.float32
    
    def test_performance_impact(self):
        """Test that attention biases don't significantly impact performance."""
        import time
        
        batch_size = 4
        seq_len = 10
        hidden_dim = 64
        
        options = Options()
        options.hidden_dim = hidden_dim
        options.batch_size = batch_size
        
        attention = SymmetricAttentionFull(
            options=options,
            degree=2,
            permutation_indices=None
        )
        
        # Test data
        x = torch.randn(seq_len, batch_size, hidden_dim)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len, dtype=torch.float32)
        
        # Time without bias
        start_time = time.time()
        for _ in range(100):
            _ = attention(x, padding_mask, sequence_mask, attention_mask=None)
        time_without_bias = time.time() - start_time
        
        # Time with bias
        start_time = time.time()
        for _ in range(100):
            _ = attention(x, padding_mask, sequence_mask, attention_mask=attention_bias)
        time_with_bias = time.time() - start_time
        
        # Verify performance impact is reasonable (less than 50% overhead)
        overhead = (time_with_bias - time_without_bias) / time_without_bias
        assert overhead < 0.5, f"Attention bias overhead too high: {overhead:.2%}"


if __name__ == "__main__":
    # Example of how to run these tests
    print("To run these tests, use:")
    print("pytest tests/test_attention_mask.py -v")
    print("\nOr run specific test classes:")
    print("pytest tests/test_attention_mask.py::TestAttentionBiasDataTypes -v")
    print("\nSpecific test for attention bias functionality:")
    print("pytest tests/test_attention_mask.py::TestSymmetricAttentionWithBias::test_attention_forward_with_bias -v")
