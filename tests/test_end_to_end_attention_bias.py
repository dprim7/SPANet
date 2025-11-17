"""
Comprehensive end-to-end tests for the multi-jet attention bias system.

These tests validate the complete data flow from HDF5 files through EventInfo,
Input classes, Source objects, and finally to the network.
"""

import os
import tempfile
import pytest
import h5py
import numpy as np
import torch
import yaml

# Test if SPANet modules are available
SPANET_AVAILABLE = True
SKIP_REASON = ""

try:
    from spanet.dataset.event_info import EventInfo
    from spanet.dataset.inputs import create_source_input
    from spanet.dataset.inputs.AttentionBiasInput import AttentionBiasInput
    from spanet.dataset.inputs.MultiJetAttentionBiasInput import MultiJetAttentionBiasInput
    from spanet.dataset.types import InputType, Source
    from spanet.network.utilities.multi_jet_attention_bias import (
        concatenate_jet_overlap_matrices,
        validate_jet_type_bias_consistency
    )
except ImportError as e:
    SPANET_AVAILABLE = False
    SKIP_REASON = f"SPANet modules not available: {e}"


@pytest.mark.skipif(not SPANET_AVAILABLE, reason=SKIP_REASON)
class TestEndToEndAttentionBiasFlow:
    """Test the complete attention bias data flow end-to-end."""

    @pytest.fixture
    def sample_event_config_single_bias(self):
        """Create a sample event configuration with single attention bias."""
        return {
            'INPUTS': {
                'SEQUENTIAL': {
                    'Source': {
                        'pt': 'log_normalize',
                        'eta': 'normalize',
                        'phi': 'normalize'
                    }
                },
                'ATTENTION_BIAS': {
                    'AKOverlap': {
                        'ak_overlap_scores': 'none'
                    }
                }
            },
            'EVENT': {
                't1': ['q1', 'q2', 'b'],
                't2': ['q1', 'q2', 'b']
            },
            'PERMUTATIONS': {
                'EVENT': [['t1', 't2']],
                't1': [['q1', 'q2']],
                't2': [['q1', 'q2']]
            },
            'REGRESSIONS': {},
            'CLASSIFICATIONS': {}
        }

    @pytest.fixture
    def sample_event_config_multi_bias(self):
        """Create a sample event configuration with multi-jet attention bias."""
        return {
            'INPUTS': {
                'SEQUENTIAL': {
                    'Source': {
                        'pt': 'log_normalize',
                        'eta': 'normalize', 
                        'phi': 'normalize'
                    }
                },
                'ATTENTION_BIAS': {
                    'MultiJetOverlap': {
                        'ak5_overlap': 'none',
                        'ak8_overlap': 'none',
                        'ak15_overlap': 'none'
                    }
                }
            },
            'EVENT': {
                't1': ['q1', 'q2', 'b'],
                't2': ['q1', 'q2', 'b']
            },
            'PERMUTATIONS': {
                'EVENT': [['t1', 't2']],
                't1': [['q1', 'q2']],
                't2': [['q1', 'q2']]
            },
            'REGRESSIONS': {},
            'CLASSIFICATIONS': {}
        }

    @pytest.fixture
    def sample_hdf5_file_single_bias(self):
        """Create HDF5 file with single attention bias data."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.close()

        num_events = 10
        max_jets = 8

        with h5py.File(temp_file.name, 'w') as f:
            # Create INPUTS group structure
            inputs = f.create_group('INPUTS')
            
            # Sequential input (Source)
            source = inputs.create_group('Source')
            source.create_dataset('MASK', data=np.random.choice([True, False], (num_events, max_jets)))
            source.create_dataset('pt', data=np.random.uniform(20, 500, (num_events, max_jets)))
            source.create_dataset('eta', data=np.random.uniform(-3, 3, (num_events, max_jets)))
            source.create_dataset('phi', data=np.random.uniform(-np.pi, np.pi, (num_events, max_jets)))
            
            # Single attention bias (AKOverlap)
            ak_overlap = inputs.create_group('AKOverlap')
            
            # Create symmetric bias matrices
            bias_data = np.zeros((num_events, max_jets, max_jets), dtype=np.float32)
            for i in range(num_events):
                # Random symmetric matrix
                matrix = np.random.normal(0, 0.5, (max_jets, max_jets)).astype(np.float32)
                matrix = (matrix + matrix.T) / 2  # Make symmetric
                np.fill_diagonal(matrix, 0.0)  # Zero diagonal
                bias_data[i] = matrix
                
            ak_overlap.create_dataset('ak_overlap_scores', data=bias_data)
            ak_overlap.create_dataset('MASK', data=np.random.choice([True, False], (num_events, max_jets)))

        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture  
    def sample_hdf5_file_multi_bias(self):
        """Create HDF5 file with multi-jet attention bias data."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.close()

        num_events = 10
        max_jets = 8
        ak5_size = 5
        ak8_size = 2  
        ak15_size = 1

        with h5py.File(temp_file.name, 'w') as f:
            # Create INPUTS group structure
            inputs = f.create_group('INPUTS')
            
            # Sequential input (Source)
            source = inputs.create_group('Source')
            source.create_dataset('MASK', data=np.random.choice([True, False], (num_events, max_jets)))
            source.create_dataset('pt', data=np.random.uniform(20, 500, (num_events, max_jets)))
            source.create_dataset('eta', data=np.random.uniform(-3, 3, (num_events, max_jets)))
            source.create_dataset('phi', data=np.random.uniform(-np.pi, np.pi, (num_events, max_jets)))
            
            # Multi-jet attention bias (MultiJetOverlap)
            multi_overlap = inputs.create_group('MultiJetOverlap')
            
            # Create separate bias matrices for each jet type
            for jet_type, size in [('ak5_overlap', ak5_size), ('ak8_overlap', ak8_size), ('ak15_overlap', ak15_size)]:
                bias_data = np.zeros((num_events, size, size), dtype=np.float32)
                for i in range(num_events):
                    matrix = np.random.normal(0, 0.3, (size, size)).astype(np.float32)
                    matrix = (matrix + matrix.T) / 2  # Make symmetric
                    np.fill_diagonal(matrix, 0.0)  # Zero diagonal
                    bias_data[i] = matrix
                    
                multi_overlap.create_dataset(jet_type, data=bias_data)
            
            # Overall mask for the combined jet sequence
            multi_overlap.create_dataset('MASK', data=np.random.choice([True, False], (num_events, max_jets)))

        yield temp_file.name
        os.unlink(temp_file.name)

    def test_single_bias_complete_flow(self, sample_event_config_single_bias, sample_hdf5_file_single_bias):
        """Test complete flow for single attention bias."""
        # Step 1: Create EventInfo from configuration
        event_info = self._create_event_info_from_config(sample_event_config_single_bias)
        
        # Verify input types are correct
        assert event_info.input_type('Source') == InputType.Sequential
        assert event_info.input_type('AKOverlap') == InputType.AttentionBias
        
        # Step 2: Load data using input factory
        with h5py.File(sample_hdf5_file_single_bias, 'r') as hdf5_file:
            num_events = 10
            limit_index = np.arange(num_events)
            
            # Create source input
            source_input = create_source_input(event_info, hdf5_file, 'Source', num_events, limit_index)
            
            # Create attention bias input - should automatically select AttentionBiasInput
            bias_input = create_source_input(event_info, hdf5_file, 'AKOverlap', num_events, limit_index)
            
            # Verify correct input class was selected
            assert isinstance(bias_input, AttentionBiasInput)
            assert not isinstance(bias_input, MultiJetAttentionBiasInput)
            
            # Step 3: Verify Source objects are created correctly
            source_obj = source_input[0]
            bias_obj = bias_input[0]
            
            assert isinstance(source_obj, Source)
            assert isinstance(bias_obj, Source)
            
            # Step 4: Verify attention bias is present and valid
            assert bias_obj.ak_overlap_mask is not None
            assert bias_obj.ak_overlap_mask.shape[0] == bias_obj.ak_overlap_mask.shape[1]  # Square matrix
            assert bias_obj.ak_overlap_mask.dtype == torch.float32
            
            # Check symmetry
            bias_matrix = bias_obj.ak_overlap_mask
            assert torch.allclose(bias_matrix, bias_matrix.T, atol=1e-6), "Bias matrix should be symmetric"
            
            # Check diagonal is zero
            diagonal = torch.diagonal(bias_matrix)
            assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6), "Diagonal should be zero"

    def test_multi_bias_complete_flow(self, sample_event_config_multi_bias, sample_hdf5_file_multi_bias):
        """Test complete flow for multi-jet attention bias."""
        # Step 1: Create EventInfo from configuration  
        event_info = self._create_event_info_from_config(sample_event_config_multi_bias)
        
        # Verify input types are correct
        assert event_info.input_type('Source') == InputType.Sequential
        assert event_info.input_type('MultiJetOverlap') == InputType.AttentionBias
        
        # Step 2: Load data using input factory
        with h5py.File(sample_hdf5_file_multi_bias, 'r') as hdf5_file:
            num_events = 10
            limit_index = np.arange(num_events)
            
            # Create source input
            source_input = create_source_input(event_info, hdf5_file, 'Source', num_events, limit_index)
            
            # Create attention bias input - should automatically select MultiJetAttentionBiasInput
            bias_input = create_source_input(event_info, hdf5_file, 'MultiJetOverlap', num_events, limit_index)
            
            # Verify correct input class was selected
            assert isinstance(bias_input, MultiJetAttentionBiasInput)
            
            # Step 3: Verify Source objects are created correctly
            source_obj = source_input[0]
            bias_obj = bias_input[0]
            
            assert isinstance(source_obj, Source)
            assert isinstance(bias_obj, Source)
            
            # Step 4: Verify multi-jet bias structure
            assert hasattr(bias_obj, 'ak5_overlap_mask')
            assert hasattr(bias_obj, 'ak8_overlap_mask') 
            assert hasattr(bias_obj, 'ak15_overlap_mask')
            
            # Check individual matrices exist and are valid
            assert bias_obj.ak5_overlap_mask is not None
            assert bias_obj.ak8_overlap_mask is not None
            assert bias_obj.ak15_overlap_mask is not None
            
            # Verify dimensions
            assert bias_obj.ak5_overlap_mask.shape == (5, 5)
            assert bias_obj.ak8_overlap_mask.shape == (2, 2)
            assert bias_obj.ak15_overlap_mask.shape == (1, 1)

    def test_attention_bias_network_integration(self, sample_event_config_single_bias, sample_hdf5_file_single_bias):
        """Test that attention bias integrates correctly with network validation."""
        from spanet.network.utilities.multi_jet_attention_bias import extract_multi_jet_attention_bias
        
        # Create EventInfo and load data
        event_info = self._create_event_info_from_config(sample_event_config_single_bias)
        
        with h5py.File(sample_hdf5_file_single_bias, 'r') as hdf5_file:
            num_events = 10
            limit_index = np.arange(num_events)
            
            source_input = create_source_input(event_info, hdf5_file, 'Source', num_events, limit_index)
            bias_input = create_source_input(event_info, hdf5_file, 'AKOverlap', num_events, limit_index)
            
            # Create sources tuple like network would
            sources = (source_input[0], bias_input[0])
            
            # Test attention bias extraction
            attention_bias = extract_multi_jet_attention_bias(sources, event_info)
            
            assert attention_bias is not None
            assert attention_bias.shape[0] == attention_bias.shape[1]  # Square
            assert attention_bias.dtype == torch.float32

    def test_input_factory_routing_logic(self, sample_event_config_single_bias, sample_event_config_multi_bias):
        """Test that input factory correctly routes to appropriate input classes."""
        # Test single bias routing
        event_info_single = self._create_event_info_from_config(sample_event_config_single_bias)
        
        # Mock HDF5 file for factory test
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            with h5py.File(temp_file.name, 'w') as f:
                inputs = f.create_group('INPUTS')
                ak_overlap = inputs.create_group('AKOverlap')
                ak_overlap.create_dataset('ak_overlap_scores', data=np.zeros((5, 4, 4)))
                
                # Test that single feature leads to AttentionBiasInput
                bias_input = create_source_input(event_info_single, f, 'AKOverlap', 5, np.arange(5))
                assert isinstance(bias_input, AttentionBiasInput)
                assert not isinstance(bias_input, MultiJetAttentionBiasInput)
        
        # Test multi bias routing
        event_info_multi = self._create_event_info_from_config(sample_event_config_multi_bias)
        
        with tempfile.NamedTemporaryFile(suffix='.h5') as temp_file:
            with h5py.File(temp_file.name, 'w') as f:
                inputs = f.create_group('INPUTS')
                multi_overlap = inputs.create_group('MultiJetOverlap')
                multi_overlap.create_dataset('ak5_overlap', data=np.zeros((5, 3, 3)))
                multi_overlap.create_dataset('ak8_overlap', data=np.zeros((5, 2, 2)))
                multi_overlap.create_dataset('ak15_overlap', data=np.zeros((5, 1, 1)))
                
                # Test that multiple jet-type features lead to MultiJetAttentionBiasInput
                bias_input = create_source_input(event_info_multi, f, 'MultiJetOverlap', 5, np.arange(5))
                assert isinstance(bias_input, MultiJetAttentionBiasInput)

    def test_validation_integration(self, sample_event_config_multi_bias, sample_hdf5_file_multi_bias):
        """Test that validation works with loaded attention bias data."""
        event_info = self._create_event_info_from_config(sample_event_config_multi_bias)
        
        with h5py.File(sample_hdf5_file_multi_bias, 'r') as hdf5_file:
            num_events = 10
            limit_index = np.arange(num_events)
            
            source_input = create_source_input(event_info, hdf5_file, 'Source', num_events, limit_index)
            bias_input = create_source_input(event_info, hdf5_file, 'MultiJetOverlap', num_events, limit_index)
            
            # Get Source objects
            source_obj = source_input[0]
            bias_obj = bias_input[0]
            
            # Create sources tuple and test validation
            sources = (source_obj, bias_obj)
            
            # Mock event_info with jet types
            mock_event_info = type('MockEventInfo', (), {
                'jet_types': ['ak5', 'ak8', 'ak15'],
                'input_types': {'Source': 'SEQUENTIAL'},
                'input_names': ['Source']
            })()
            
            # This should not raise an error
            validate_jet_type_bias_consistency(sources, mock_event_info)

    def _create_event_info_from_config(self, config):
        """Helper to create EventInfo from configuration dictionary."""
        # Create temporary YAML file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_file)
        temp_file.close()
        
        try:
            event_info = EventInfo.read_from_yaml(temp_file.name)
            return event_info
        finally:
            os.unlink(temp_file.name)


@pytest.mark.skipif(not SPANET_AVAILABLE, reason=SKIP_REASON)
class TestAttentionBiasErrorHandling:
    """Test error handling and edge cases for attention bias system."""

    def test_missing_bias_data_handling(self):
        """Test graceful handling when bias data is missing."""
        config = {
            'INPUTS': {
                'SEQUENTIAL': {'Source': {'pt': 'none'}},
                'ATTENTION_BIAS': {'MissingBias': {'nonexistent_feature': 'none'}}
            },
            'EVENT': {'t1': ['q1']},
            'PERMUTATIONS': {'EVENT': []},
            'REGRESSIONS': {},
            'CLASSIFICATIONS': {}
        }
        
        # Create temporary files
        temp_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_yaml)
        temp_yaml.close()
        
        temp_hdf5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        temp_hdf5.close()
        
        try:
            # Create HDF5 with missing bias data
            with h5py.File(temp_hdf5.name, 'w') as f:
                inputs = f.create_group('INPUTS')
                source = inputs.create_group('Source')
                source.create_dataset('pt', data=np.random.rand(5, 4))
                source.create_dataset('MASK', data=np.ones((5, 4), dtype=bool))
                # Note: MissingBias group is intentionally missing
            
            event_info = EventInfo.read_from_yaml(temp_yaml.name)
            
            # This should raise an appropriate error
            with pytest.raises((KeyError, ValueError)):
                with h5py.File(temp_hdf5.name, 'r') as f:
                    create_source_input(event_info, f, 'MissingBias', 5, np.arange(5))
                    
        finally:
            os.unlink(temp_yaml.name)
            os.unlink(temp_hdf5.name)

    def test_inconsistent_jet_types_validation(self):
        """Test validation catches inconsistent jet type configurations."""
        # Create sources with insufficient bias matrices
        source1 = Source(
            data=torch.randn(2, 4, 3),
            mask=torch.ones(2, 4, dtype=torch.bool),
            ak5_overlap_mask=torch.randn(2, 3, 3),  # AK5 only
        )
        
        source2 = Source(
            data=torch.randn(2, 4, 3), 
            mask=torch.ones(2, 4, dtype=torch.bool),
            # No bias matrices at all
        )
        
        sources = (source1, source2)
        
        # Mock event_info expecting multiple jet types in the sequential inputs
        mock_event_info = type('MockEventInfo', (), {
            'input_types': {
                'AK5Jets': 'SEQUENTIAL',  # Expects AK5 bias
                'AK8Jets': 'SEQUENTIAL',  # Expects AK8 bias (but missing)
                'AK15Jets': 'SEQUENTIAL'  # Expects AK15 bias (but missing)
            },
            'input_names': ['AK5Jets', 'AK8Jets', 'AK15Jets']
        })()
        
        # This should raise an assertion error because we only have 1 bias matrix but expect 3
        with pytest.raises(AssertionError, match="Mismatch between jet types and bias matrices"):
            validate_jet_type_bias_consistency(sources, mock_event_info)


if __name__ == "__main__":
    pytest.main([__file__])
