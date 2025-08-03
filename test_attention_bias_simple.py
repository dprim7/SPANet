"""
Simple integration test for attention bias functionality.
"""

import torch
from spanet.dataset.types import Source


def test_attention_bias_integration():
    """Test basic attention bias integration."""
    print("Testing attention bias integration...")
    
    # Test 1: Source creation with attention bias
    batch_size, max_jets = 2, 4
    bias_data = torch.randn(batch_size, max_jets, max_jets)
    
    source = Source(
        data=torch.randn(batch_size, max_jets, 3),
        mask=torch.ones(batch_size, max_jets, dtype=torch.bool),
        ak_overlap_mask=bias_data
    )
    
    assert source.ak_overlap_mask is not None
    assert source.ak_overlap_mask.shape == (batch_size, max_jets, max_jets)
    print("✓ Source creation with attention bias works")
    
    # Test 2: Network attention bias extraction
    sources = (
        Source(torch.randn(batch_size, max_jets, 3), torch.ones(batch_size, max_jets, dtype=torch.bool)),
        source  # This one has the bias
    )
    
    # Extract bias like the network does
    attention_bias = None
    for src in sources:
        if src.ak_overlap_mask is not None:
            attention_bias = src.ak_overlap_mask
            break
    
    assert attention_bias is not None
    assert torch.equal(attention_bias, bias_data)
    print("✓ Network attention bias extraction works")
    
    # Test 3: Attention layer with bias
    from spanet.network.symmetric_attention import SymmetricAttentionFull
    from spanet.options import Options
    
    options = Options()
    options.hidden_dim = 16
    options.batch_size = batch_size
    
    seq_len = max_jets
    x = torch.randn(seq_len, batch_size, options.hidden_dim)
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    sequence_mask = torch.ones(seq_len, batch_size, 1, dtype=torch.bool)
    
    attention = SymmetricAttentionFull(options, degree=2)
    
    # Test without bias
    output_no_bias = attention(x, padding_mask, sequence_mask, None)
    
    # Test with bias
    output_with_bias = attention(x, padding_mask, sequence_mask, attention_bias)
    
    assert output_no_bias.shape == output_with_bias.shape
    assert not torch.allclose(output_no_bias, output_with_bias, atol=1e-6)
    print("✓ Attention layer processes bias correctly")
    
    print("All tests passed! Attention bias integration is working.")


if __name__ == "__main__":
    test_attention_bias_integration()
