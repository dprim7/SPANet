"""
Tests for multi-jet-type attention bias functionality.

This module tests the new multi-matrix attention bias system that handles
separate overlap matrices for AK5, AK8, and AK15 jets.
"""

import pytest
import torch
import numpy as np
import h5py
import tempfile
import os

from spanet.dataset.types import Source
from spanet.network.utilities.multi_jet_attention_bias import (
    concatenate_jet_overlap_matrices,
    extract_multi_jet_attention_bias,
    validate_overlap_matrix_symmetry,
    get_jet_type_indices,
    validate_jet_type_bias_consistency
)
from spanet.dataset.inputs.MultiJetAttentionBiasInput import MultiJetAttentionBiasInput


class TestMultiJetAttentionBias:
    """Test multi-jet-type attention bias concatenation."""
    
    def test_concatenate_jet_overlap_matrices_all_types(self):
        """Test concatenation with all three jet types."""
        batch_size = 2
        ak5_size, ak8_size, ak15_size = 10, 2, 2
        
        # Create test matrices
        ak5_overlap = torch.randn(batch_size, ak5_size, ak5_size)
        ak8_overlap = torch.randn(batch_size, ak8_size, ak8_size)
        ak15_overlap = torch.randn(batch_size, ak15_size, ak15_size)
        
        # Make them symmetric
        ak5_overlap = (ak5_overlap + ak5_overlap.transpose(-2, -1)) / 2
        ak8_overlap = (ak8_overlap + ak8_overlap.transpose(-2, -1)) / 2
        ak15_overlap = (ak15_overlap + ak15_overlap.transpose(-2, -1)) / 2
        
        combined = concatenate_jet_overlap_matrices(ak5_overlap, ak8_overlap, ak15_overlap)
        
        # Check shape
        total_size = ak5_size + ak8_size + ak15_size
        assert combined.shape == (batch_size, total_size, total_size)
        
        # Check block structure
        assert torch.equal(combined[:, :ak5_size, :ak5_size], ak5_overlap)
        assert torch.equal(combined[:, ak5_size:ak5_size+ak8_size, ak5_size:ak5_size+ak8_size], ak8_overlap)
        assert torch.equal(combined[:, ak5_size+ak8_size:, ak5_size+ak8_size:], ak15_overlap)
        
        # Check off-diagonal blocks are zero
        assert torch.allclose(combined[:, :ak5_size, ak5_size:], torch.zeros(batch_size, ak5_size, ak8_size + ak15_size))
        assert torch.allclose(combined[:, ak5_size:, :ak5_size], torch.zeros(batch_size, ak8_size + ak15_size, ak5_size))

    def test_concatenate_jet_overlap_matrices_partial(self):
        """Test concatenation with only some jet types."""
        batch_size = 2
        ak5_size, ak8_size, ak15_size = 10, 2, 2
        
        # Only AK5 and AK15
        ak5_overlap = torch.randn(batch_size, ak5_size, ak5_size)
        ak15_overlap = torch.randn(batch_size, ak15_size, ak15_size)
        
        combined = concatenate_jet_overlap_matrices(ak5_overlap, None, ak15_overlap)
        
        total_size = ak5_size + ak8_size + ak15_size
        assert combined.shape == (batch_size, total_size, total_size)
        
        # Check AK5 block
        assert torch.equal(combined[:, :ak5_size, :ak5_size], ak5_overlap)
        
        # Check AK8 block is zero (not provided)
        ak8_slice = slice(ak5_size, ak5_size + ak8_size)
        assert torch.allclose(combined[:, ak8_slice, ak8_slice], torch.zeros(batch_size, ak8_size, ak8_size))
        
        # Check AK15 block
        ak15_slice = slice(ak5_size + ak8_size, total_size)
        assert torch.equal(combined[:, ak15_slice, ak15_slice], ak15_overlap)

    def test_concatenate_jet_overlap_matrices_empty(self):
        """Test concatenation with no matrices."""
        combined = concatenate_jet_overlap_matrices(None, None, None)
        assert combined is None

    def test_extract_multi_jet_attention_bias_legacy(self):
        """Test extraction with legacy single overlap matrix."""
        batch_size = 2
        total_size = 14  # 10 + 2 + 2
        
        # Create legacy source with single overlap matrix
        legacy_overlap = torch.randn(batch_size, total_size, total_size)
        legacy_source = Source(
            data=torch.randn(batch_size, total_size, 3),
            mask=torch.ones(batch_size, total_size, dtype=torch.bool),
            ak_overlap_mask=legacy_overlap
        )
        
        sources = (legacy_source,)
        extracted_bias = extract_multi_jet_attention_bias(sources)
        
        assert torch.equal(extracted_bias, legacy_overlap)

    def test_extract_multi_jet_attention_bias_multi_type(self):
        """Test extraction with multi-jet-type overlap matrices."""
        batch_size = 2
        ak5_size, ak8_size, ak15_size = 10, 2, 2
        
        # Create individual jet type overlap matrices
        ak5_overlap = torch.randn(batch_size, ak5_size, ak5_size)
        ak8_overlap = torch.randn(batch_size, ak8_size, ak8_size)
        ak15_overlap = torch.randn(batch_size, ak15_size, ak15_size)
        
        # Create sources with individual matrices
        ak5_source = Source(
            data=torch.randn(batch_size, ak5_size, 3),
            mask=torch.ones(batch_size, ak5_size, dtype=torch.bool),
            ak5_overlap_mask=ak5_overlap
        )
        
        ak8_source = Source(
            data=torch.randn(batch_size, ak8_size, 3),
            mask=torch.ones(batch_size, ak8_size, dtype=torch.bool),
            ak8_overlap_mask=ak8_overlap
        )
        
        ak15_source = Source(
            data=torch.randn(batch_size, ak15_size, 3),
            mask=torch.ones(batch_size, ak15_size, dtype=torch.bool),
            ak15_overlap_mask=ak15_overlap
        )
        
        sources = (ak5_source, ak8_source, ak15_source)
        extracted_bias = extract_multi_jet_attention_bias(sources)
        
        # Should get combined matrix
        total_size = ak5_size + ak8_size + ak15_size
        assert extracted_bias.shape == (batch_size, total_size, total_size)
        
        # Check that individual blocks match
        assert torch.equal(extracted_bias[:, :ak5_size, :ak5_size], ak5_overlap)
        assert torch.equal(extracted_bias[:, ak5_size:ak5_size+ak8_size, ak5_size:ak5_size+ak8_size], ak8_overlap)
        assert torch.equal(extracted_bias[:, ak5_size+ak8_size:, ak5_size+ak8_size:], ak15_overlap)

    def test_validate_overlap_matrix_symmetry(self):
        """Test symmetry validation function."""
        # Symmetric 2D matrix
        symmetric_2d = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        assert validate_overlap_matrix_symmetry(symmetric_2d)
        
        # Asymmetric 2D matrix  
        asymmetric_2d = torch.tensor([[1.0, 0.5], [0.3, 1.0]])
        assert not validate_overlap_matrix_symmetry(asymmetric_2d)
        
        # Symmetric 3D matrix (batch)
        symmetric_3d = torch.stack([symmetric_2d, symmetric_2d])
        assert validate_overlap_matrix_symmetry(symmetric_3d)
        
        # Asymmetric 3D matrix (batch)
        asymmetric_3d = torch.stack([symmetric_2d, asymmetric_2d])
        assert not validate_overlap_matrix_symmetry(asymmetric_3d)

    def test_get_jet_type_indices(self):
        """Test jet type index calculation."""
        ak5_slice, ak8_slice, ak15_slice = get_jet_type_indices()
        
        assert ak5_slice == slice(0, 10)
        assert ak8_slice == slice(10, 12)
        assert ak15_slice == slice(12, 14)
        
        # Test custom sizes
        ak5_slice, ak8_slice, ak15_slice = get_jet_type_indices(ak5_size=5, ak8_size=3, ak15_size=1)
        
        assert ak5_slice == slice(0, 5)
        assert ak8_slice == slice(5, 8)
        assert ak15_slice == slice(8, 9)

    def test_validate_jet_type_bias_consistency(self):
        """Test validation of jet types vs bias matrices consistency."""
        from spanet.network.utilities.multi_jet_attention_bias import validate_jet_type_bias_consistency
        
        # Mock event info with multiple jet types
        class MockEventInfo:
            def __init__(self, input_types):
                self.input_types = input_types
        
        # Test case 1: Multiple jet types defined, all bias matrices present
        event_info = MockEventInfo({
            'AK5Jets': 'SEQUENTIAL',
            'AK8Jets': 'SEQUENTIAL', 
            'AK15Jets': 'SEQUENTIAL'
        })
        
        # Create sources with all required bias matrices
        ak5_source = Source(
            data=torch.randn(2, 10, 3),
            mask=torch.ones(2, 10, dtype=torch.bool),
            ak5_overlap_mask=torch.randn(2, 10, 10)
        )
        
        ak8_source = Source(
            data=torch.randn(2, 2, 3),
            mask=torch.ones(2, 2, dtype=torch.bool),
            ak8_overlap_mask=torch.randn(2, 2, 2)
        )
        
        ak15_source = Source(
            data=torch.randn(2, 2, 3),
            mask=torch.ones(2, 2, dtype=torch.bool),
            ak15_overlap_mask=torch.randn(2, 2, 2)
        )
        
        sources = (ak5_source, ak8_source, ak15_source)
        
        # Should not raise an assertion error
        validate_jet_type_bias_consistency(sources, event_info)
        
        # Test case 2: Multiple jet types defined, missing bias matrix
        sources_missing = (ak5_source, ak8_source)  # Missing AK15
        
        with pytest.raises(AssertionError, match="Mismatch between jet types and bias matrices"):
            validate_jet_type_bias_consistency(sources_missing, event_info)
        
        # Test case 3: Specific jet type missing (should trigger count mismatch first)
        sources_missing_specific = (
            ak5_source,
            Source(  # AK8 source without bias matrix
                data=torch.randn(2, 2, 3),
                mask=torch.ones(2, 2, dtype=torch.bool)
            ),
            ak15_source
        )
        
        with pytest.raises(AssertionError, match="Mismatch between jet types and bias matrices"):
            validate_jet_type_bias_consistency(sources_missing_specific, event_info)
        
        # Test case 4: Test specific jet type missing error (with correct count)
        # Create a scenario where count matches but specific type is wrong
        sources_wrong_type = (
            ak5_source,
            ak5_source,  # Wrong type (should be AK8)
            ak15_source
        )
        
        with pytest.raises(AssertionError, match="Missing bias matrix for jet type"):
            validate_jet_type_bias_consistency(sources_wrong_type, event_info)
        
        # Test case 5: Single jet type (should not trigger validation)
        single_jet_event_info = MockEventInfo({'AK5Jets': 'SEQUENTIAL'})
        single_sources = (ak5_source,)
        
        # Should pass even if other bias matrices are missing
        validate_jet_type_bias_consistency(single_sources, single_jet_event_info)
        
        # Test case 6: No sequential inputs (should pass)
        no_jets_event_info = MockEventInfo({'GlobalFeatures': 'GLOBAL'})
        empty_sources = ()
        
        validate_jet_type_bias_consistency(empty_sources, no_jets_event_info)


class TestMultiJetAttentionBiasInput:
    """Test the MultiJetAttentionBiasInput class."""
    
    @pytest.fixture
    def sample_multi_jet_hdf5_file(self):
        """Create a temporary HDF5 file with multi-jet attention bias data."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        num_events = 100
        ak5_size, ak8_size, ak15_size = 10, 2, 2
        
        with h5py.File(temp_file.name, 'w') as f:
            # Create input group
            jets_group = f.create_group('INPUTS/Jets')
            
            # Create AK5 overlap matrix
            ak5_overlaps = np.random.randn(num_events, ak5_size, ak5_size).astype(np.float32)
            for i in range(num_events):
                # Make symmetric
                ak5_overlaps[i] = (ak5_overlaps[i] + ak5_overlaps[i].T) / 2
                # Set diagonal to 0
                np.fill_diagonal(ak5_overlaps[i], 0.0)
            jets_group.create_dataset('ak5_overlap', data=ak5_overlaps)
            
            # Create AK8 overlap matrix
            ak8_overlaps = np.random.randn(num_events, ak8_size, ak8_size).astype(np.float32)
            for i in range(num_events):
                ak8_overlaps[i] = (ak8_overlaps[i] + ak8_overlaps[i].T) / 2
                np.fill_diagonal(ak8_overlaps[i], 0.0)
            jets_group.create_dataset('ak8_overlap', data=ak8_overlaps)
            
            # Create AK15 overlap matrix
            ak15_overlaps = np.random.randn(num_events, ak15_size, ak15_size).astype(np.float32)
            for i in range(num_events):
                ak15_overlaps[i] = (ak15_overlaps[i] + ak15_overlaps[i].T) / 2
                np.fill_diagonal(ak15_overlaps[i], 0.0)
            jets_group.create_dataset('ak15_overlap', data=ak15_overlaps)
            
            # Create masks (all jets valid for simplicity)
            ak5_mask = np.ones((num_events, ak5_size), dtype=bool)
            ak8_mask = np.ones((num_events, ak8_size), dtype=bool)
            ak15_mask = np.ones((num_events, ak15_size), dtype=bool)
            
            jets_group.create_dataset('ak5_overlap_mask', data=ak5_mask)
            jets_group.create_dataset('ak8_overlap_mask', data=ak8_mask)
            jets_group.create_dataset('ak15_overlap_mask', data=ak15_mask)
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)

    def test_multi_jet_hdf5_loading(self, sample_multi_jet_hdf5_file):
        """Test loading multi-jet bias data from HDF5."""
        # Create a mock event info
        from spanet.dataset.types import FeatureInfo
        
        class MockEventInfo:
            def __init__(self):
                self.input_features = {
                    "Jets": [
                        FeatureInfo('ak5_overlap', normalize=False, log_scale=False),
                        FeatureInfo('ak8_overlap', normalize=False, log_scale=False),
                        FeatureInfo('ak15_overlap', normalize=False, log_scale=False)
                    ]
                }
        
        # Create input loader with proper arguments
        with h5py.File(sample_multi_jet_hdf5_file, 'r') as f:
            event_info = MockEventInfo()
            bias_input = MultiJetAttentionBiasInput(
                event_info=event_info,
                hdf5_file=f,
                input_name="Jets",
                num_events=100,
                limit_index=np.arange(100)
            )
        
        # Check that data was loaded
        assert bias_input.ak5_bias_data is not None
        assert bias_input.ak8_bias_data is not None
        assert bias_input.ak15_bias_data is not None
        assert bias_input.combined_bias_data is not None
        
        # Check shapes
        assert bias_input.ak5_bias_data.shape == (100, 10, 10)
        assert bias_input.ak8_bias_data.shape == (100, 2, 2)
        assert bias_input.ak15_bias_data.shape == (100, 2, 2)
        assert bias_input.combined_bias_data.shape == (100, 14, 14)
        
        # Check block structure in combined matrix
        combined = bias_input.combined_bias_data
        assert torch.equal(combined[:, :10, :10], bias_input.ak5_bias_data)
        assert torch.equal(combined[:, 10:12, 10:12], bias_input.ak8_bias_data)
        assert torch.equal(combined[:, 12:14, 12:14], bias_input.ak15_bias_data)

    def test_multi_jet_source_creation(self, sample_multi_jet_hdf5_file):
        """Test Source creation with multi-jet bias data."""
        # Create a mock event info
        from spanet.dataset.types import FeatureInfo
        
        class MockEventInfo:
            def __init__(self):
                self.input_features = {
                    "Jets": [
                        FeatureInfo('ak5_overlap', normalize=False, log_scale=False),
                        FeatureInfo('ak8_overlap', normalize=False, log_scale=False),
                        FeatureInfo('ak15_overlap', normalize=False, log_scale=False)
                    ]
                }
        
        # Create and load bias input
        with h5py.File(sample_multi_jet_hdf5_file, 'r') as f:
            event_info = MockEventInfo()
            bias_input = MultiJetAttentionBiasInput(
                event_info=event_info,
                hdf5_file=f,
                input_name="Jets",
                num_events=100,
                limit_index=np.arange(100)
            )
        
        # Get a Source for the first event
        source = bias_input[0]
        
        # Check Source structure
        assert isinstance(source, Source)
        assert source.ak_overlap_mask is not None  # Combined matrix (backward compatibility)
        assert source.ak5_overlap_mask is not None  # Individual AK5 matrix
        assert source.ak8_overlap_mask is not None  # Individual AK8 matrix
        assert source.ak15_overlap_mask is not None  # Individual AK15 matrix
        
        # Check shapes
        assert source.ak_overlap_mask.shape == (14, 14)  # Combined
        assert source.ak5_overlap_mask.shape == (10, 10)  # AK5
        assert source.ak8_overlap_mask.shape == (2, 2)    # AK8
        assert source.ak15_overlap_mask.shape == (2, 2)   # AK15


class TestEndToEndMultiJetIntegration:
    """Test end-to-end integration of multi-jet attention bias."""
    
    def test_multi_jet_bias_propagation(self):
        """Test that multi-jet biases propagate correctly through the extraction pipeline."""
        batch_size = 2
        ak5_size, ak8_size, ak15_size = 10, 2, 2
        
        # Create individual jet type overlap matrices
        ak5_overlap = torch.randn(batch_size, ak5_size, ak5_size)
        ak8_overlap = torch.randn(batch_size, ak8_size, ak8_size)
        ak15_overlap = torch.randn(batch_size, ak15_size, ak15_size)
        
        # Make them symmetric
        ak5_overlap = (ak5_overlap + ak5_overlap.transpose(-2, -1)) / 2
        ak8_overlap = (ak8_overlap + ak8_overlap.transpose(-2, -1)) / 2
        ak15_overlap = (ak15_overlap + ak15_overlap.transpose(-2, -1)) / 2
        
        # Create sources with multi-jet bias data
        sources = []
        
        # AK5 source
        ak5_source = Source(
            data=torch.randn(batch_size, ak5_size, 3),
            mask=torch.ones(batch_size, ak5_size, dtype=torch.bool),
            ak5_overlap_mask=ak5_overlap
        )
        sources.append(ak5_source)
        
        # AK8 source
        ak8_source = Source(
            data=torch.randn(batch_size, ak8_size, 3),
            mask=torch.ones(batch_size, ak8_size, dtype=torch.bool),
            ak8_overlap_mask=ak8_overlap
        )
        sources.append(ak8_source)
        
        # AK15 source
        ak15_source = Source(
            data=torch.randn(batch_size, ak15_size, 3),
            mask=torch.ones(batch_size, ak15_size, dtype=torch.bool),
            ak15_overlap_mask=ak15_overlap
        )
        sources.append(ak15_source)
        
        # Extract combined bias
        combined_bias = extract_multi_jet_attention_bias(tuple(sources))
        
        # Verify the combined bias has the correct structure
        total_size = ak5_size + ak8_size + ak15_size
        assert combined_bias.shape == (batch_size, total_size, total_size)
        
        # Verify symmetry is preserved
        assert validate_overlap_matrix_symmetry(combined_bias)
        
        # Verify block structure
        assert torch.equal(combined_bias[:, :ak5_size, :ak5_size], ak5_overlap)
        assert torch.equal(combined_bias[:, ak5_size:ak5_size+ak8_size, ak5_size:ak5_size+ak8_size], ak8_overlap)
        assert torch.equal(combined_bias[:, ak5_size+ak8_size:, ak5_size+ak8_size:], ak15_overlap)
        
    def test_mixed_legacy_and_multi_jet(self):
        """Test that system gracefully handles mix of legacy and new multi-jet bias."""
        batch_size = 2
        total_size = 14
        
        # Create legacy source (should take precedence)
        legacy_overlap = torch.randn(batch_size, total_size, total_size)
        legacy_source = Source(
            data=torch.randn(batch_size, total_size, 3),
            mask=torch.ones(batch_size, total_size, dtype=torch.bool),
            ak_overlap_mask=legacy_overlap
        )
        
        # Create multi-jet source
        ak5_overlap = torch.randn(batch_size, 10, 10)
        multi_jet_source = Source(
            data=torch.randn(batch_size, 10, 3),
            mask=torch.ones(batch_size, 10, dtype=torch.bool),
            ak5_overlap_mask=ak5_overlap
        )
        
        sources = (legacy_source, multi_jet_source)
        extracted_bias = extract_multi_jet_attention_bias(sources)
        
        # Should return legacy matrix (precedence)
        assert torch.equal(extracted_bias, legacy_overlap)
        
        print("✓ Legacy and multi-jet bias mixing works correctly")
