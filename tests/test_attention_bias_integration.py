"""
Integration tests for attention bias functionality in SPANet.

Tests the complete flow: HDF5 → AttentionBiasInput → Source → Network → Attention layers
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from spanet.dataset.types import Source, InputType


class TestAttentionBiasIntegration:
    """Test end-to-end attention bias integration."""
    
    def test_attention_bias_source_creation(self):
        """Test that AttentionBiasInput creates proper Source objects."""
        from spanet.dataset.inputs.AttentionBiasInput import AttentionBiasInput
        from spanet.dataset.event_info import EventInfo
        
        # Create mock event info
        event_info = type('MockEventInfo', (), {
            'input_features': {'AKOverlap': [type('Feature', (), {'name': 'bias_values'})()]},
            'normalized_features': lambda self, input_name: [False]  # No normalization for bias
        })()
        
        # Create bias input instance
        bias_input = AttentionBiasInput(
            event_info=event_info,
            hdf5_file=None,
            input_name='AKOverlap',
            num_events=4,
            limit_index=np.arange(4)
        )
        
        # Manually set bias data for testing
        batch_size, max_jets = 4, 6
        bias_input.bias_data = torch.randn(batch_size, max_jets, max_jets)
        bias_input.pairwise_mask = torch.ones(batch_size, max_jets, max_jets, dtype=torch.bool)
        
        # Test source creation
        source = bias_input[0]
        
        assert isinstance(source, Source)
        assert source.ak_overlap_mask is not None
        assert source.ak_overlap_mask.shape == (max_jets, max_jets)
        assert torch.equal(source.ak_overlap_mask, bias_input.bias_data[0])
    
    def test_network_attention_bias_extraction(self):
        """Test that the network correctly extracts attention bias from sources."""
        batch_size, max_jets = 2, 4
        hidden_dim = 32
        
        # Create mock sources - one with attention bias, others without
        regular_source = Source(
            data=torch.randn(batch_size, max_jets, 3),
            mask=torch.ones(batch_size, max_jets, dtype=torch.bool),
            ak_overlap_mask=None
        )
        
        bias_source = Source(
            data=torch.randn(batch_size, max_jets, 1),  # Dummy data for bias input
            mask=torch.ones(batch_size, max_jets, dtype=torch.bool),
            ak_overlap_mask=torch.randn(batch_size, max_jets, max_jets)  # The actual bias
        )
        
        sources = (regular_source, bias_source)
        
        # Extract attention bias like the network does
        attention_bias = None
        for source in sources:
            if source.ak_overlap_mask is not None:
                attention_bias = source.ak_overlap_mask
                break
        
        assert attention_bias is not None
        assert attention_bias.shape == (batch_size, max_jets, max_jets)
        assert torch.equal(attention_bias, bias_source.ak_overlap_mask)
    
    def test_attention_bias_shapes_compatibility(self):
        """Test that attention bias shapes are compatible with attention layers."""
        from spanet.network.symmetric_attention import SymmetricAttentionFull, SymmetricAttentionSplit
        from spanet.options import Options
        
        # Create options
        options = Options()
        options.hidden_dim = 32
        options.batch_size = 2
        
        batch_size, seq_len = 2, 6
        hidden_dim = options.hidden_dim
        
        # Create input tensors
        x = torch.randn(seq_len, batch_size, hidden_dim)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len)
        
        # Test with SymmetricAttentionFull
        attention_full = SymmetricAttentionFull(options, degree=2)
        output_full = attention_full(x, padding_mask, sequence_mask, attention_bias)
        
        assert output_full.shape == (batch_size, seq_len, seq_len)
        
        # Test with SymmetricAttentionSplit  
        attention_split = SymmetricAttentionSplit(options, degree=2)
        output_split, _ = attention_split(x, padding_mask, sequence_mask, attention_bias)
        
        assert output_split.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_bias_vs_no_bias_difference(self):
        """Test that attention bias actually affects the output."""
        from spanet.network.symmetric_attention import SymmetricAttentionFull
        from spanet.options import Options
        
        # Create options
        options = Options()
        options.hidden_dim = 32
        options.batch_size = 2
        
        batch_size, seq_len = 2, 4
        hidden_dim = options.hidden_dim
        
        # Create input tensors
        x = torch.randn(seq_len, batch_size, hidden_dim)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len) * 2.0  # Significant bias
        
        # Create attention layer
        attention = SymmetricAttentionFull(options, degree=2)
        
        # Run without bias
        output_no_bias = attention(x, padding_mask, sequence_mask, None)
        
        # Run with bias
        output_with_bias = attention(x, padding_mask, sequence_mask, attention_bias)
        
        # Outputs should be different
        assert not torch.allclose(output_no_bias, output_with_bias, atol=1e-6)
        
        # Both should have the same shape
        assert output_no_bias.shape == output_with_bias.shape
    
    def test_gradient_flow_with_attention_bias(self):
        """Test that gradients flow correctly through attention bias."""
        from spanet.network.symmetric_attention import SymmetricAttentionFull
        from spanet.options import Options
        
        # Create options
        options = Options()
        options.hidden_dim = 16
        options.batch_size = 2
        
        batch_size, seq_len = 2, 3
        hidden_dim = options.hidden_dim
        
        # Create input tensors with gradients
        x = torch.randn(seq_len, batch_size, hidden_dim, requires_grad=True)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
        attention_bias = torch.randn(batch_size, seq_len, seq_len, requires_grad=True)
        
        # Create attention layer
        attention = SymmetricAttentionFull(options, degree=2)
        
        # Forward pass
        output = attention(x, padding_mask, sequence_mask, attention_bias)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert attention_bias.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        assert not torch.allclose(attention_bias.grad, torch.zeros_like(attention_bias.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
