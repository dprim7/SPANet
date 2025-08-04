"""
Utilities for handling multi-jet-type attention bias concatenation.

This module provides functions to combine multiple overlap matrices from different 
jet types (AK5, AK8, AK15) into a single block-diagonal attention bias matrix.
"""

import torch
from torch import Tensor
from typing import List, Optional, Tuple
from spanet.dataset.types import Source


def concatenate_jet_overlap_matrices(
    ak5_overlap: Optional[Tensor] = None,
    ak8_overlap: Optional[Tensor] = None, 
    ak15_overlap: Optional[Tensor] = None,
    ak5_size: int = 10,
    ak8_size: int = 2,
    ak15_size: int = 2
) -> Optional[Tensor]:
    """
    Concatenate multiple jet overlap matrices into a single block-diagonal attention bias matrix.
    
    Args:
        ak5_overlap: AK5 jet overlap matrix [B, ak5_size, ak5_size] 
        ak8_overlap: AK8 jet overlap matrix [B, ak8_size, ak8_size]
        ak15_overlap: AK15 jet overlap matrix [B, ak15_size, ak15_size]
        ak5_size: Number of AK5 jets (default: 10)
        ak8_size: Number of AK8 jets (default: 2) 
        ak15_size: Number of AK15 jets (default: 2)
        
    Returns:
        Combined attention bias matrix [B, total_size, total_size] where total_size = ak5_size + ak8_size + ak15_size
        Returns None if no overlap matrices are provided.
    """
    # Check if any overlap matrices are provided
    overlap_matrices = [ak5_overlap, ak8_overlap, ak15_overlap]
    available_matrices = [matrix for matrix in overlap_matrices if matrix is not None]
    
    if not available_matrices:
        return None
    
    # Get batch size from the first available matrix
    batch_size = available_matrices[0].shape[0]
    total_size = ak5_size + ak8_size + ak15_size
    
    # Get device and dtype from the first available matrix
    device = available_matrices[0].device
    dtype = available_matrices[0].dtype
    
    # Initialize the combined matrix with zeros
    combined_matrix = torch.zeros(batch_size, total_size, total_size, dtype=dtype, device=device)
    
    # Fill in the block-diagonal structure
    current_offset = 0
    
    # AK5 block
    if ak5_overlap is not None:
        end_offset = current_offset + ak5_size
        combined_matrix[:, current_offset:end_offset, current_offset:end_offset] = ak5_overlap
    current_offset += ak5_size
    
    # AK8 block  
    if ak8_overlap is not None:
        end_offset = current_offset + ak8_size
        combined_matrix[:, current_offset:end_offset, current_offset:end_offset] = ak8_overlap
    current_offset += ak8_size
    
    # AK15 block
    if ak15_overlap is not None:
        end_offset = current_offset + ak15_size  
        combined_matrix[:, current_offset:end_offset, current_offset:end_offset] = ak15_overlap
    
    return combined_matrix


def extract_multi_jet_attention_bias(sources: Tuple[Source, ...], event_info=None) -> Optional[Tensor]:
    """
    Extract and combine attention bias from multiple sources with different jet types.
    
    This function looks for multiple overlap matrices in the sources and combines them
    into a single block-diagonal attention bias matrix that matches the concatenated
    jet sequence.
    
    Args:
        sources: Tuple of Source objects that may contain overlap matrices
        event_info: Optional EventInfo object for validation
        
    Returns:
        Combined attention bias matrix [B, total_jets, total_jets] or None if no bias found
    """
    # Validate consistency between jet types and bias matrices if event_info provided
    if event_info is not None:
        validate_jet_type_bias_consistency(sources, event_info)
    
    # First check for the legacy single overlap matrix
    for source in sources:
        if source.ak_overlap_mask is not None:
            return source.ak_overlap_mask
    
    # Look for multi-jet-type overlap matrices 
    ak5_overlap = None
    ak8_overlap = None
    ak15_overlap = None
    
    for source in sources:
        if source.ak5_overlap_mask is not None:
            ak5_overlap = source.ak5_overlap_mask
        if source.ak8_overlap_mask is not None:
            ak8_overlap = source.ak8_overlap_mask  
        if source.ak15_overlap_mask is not None:
            ak15_overlap = source.ak15_overlap_mask
    
    # Combine the matrices if any are found
    return concatenate_jet_overlap_matrices(ak5_overlap, ak8_overlap, ak15_overlap)


def validate_jet_type_bias_consistency(sources: Tuple[Source, ...], event_info) -> None:
    """
    Validate that for every jet type defined in the event configuration,
    there is a corresponding attention bias matrix in the sources.
    
    Args:
        sources: Tuple of Source objects
        event_info: EventInfo object with input type definitions
        
    Raises:
        AssertionError: If jet types and bias matrices don't match
    """
    # Get jet input types from event configuration
    sequential_inputs = []
    for input_name, input_type in event_info.input_types.items():
        if input_type.upper() == "SEQUENTIAL":
            sequential_inputs.append(input_name)
    
    # Map common jet input names to expected bias field names
    jet_type_mapping = {
        'AK5Jets': 'ak5_overlap_mask',
        'AK8Jets': 'ak8_overlap_mask', 
        'AK15Jets': 'ak15_overlap_mask',
        'AK5': 'ak5_overlap_mask',
        'AK8': 'ak8_overlap_mask',
        'AK15': 'ak15_overlap_mask',
        # Add more mappings as needed
    }
    
    # Count expected jet types and available bias matrices
    expected_jet_types = []
    for input_name in sequential_inputs:
        if input_name in jet_type_mapping:
            expected_jet_types.append(input_name)
    
    # Count available bias matrices
    available_bias_matrices = []
    for source in sources:
        if source.ak5_overlap_mask is not None:
            available_bias_matrices.append('AK5')
        if source.ak8_overlap_mask is not None:
            available_bias_matrices.append('AK8')
        if source.ak15_overlap_mask is not None:
            available_bias_matrices.append('AK15')
    
    # If we have multiple jet types defined, ensure we have corresponding bias matrices
    if len(expected_jet_types) > 1:
        expected_count = len(expected_jet_types)
        available_count = len(available_bias_matrices)
        
        assert available_count == expected_count, (
            f"Mismatch between jet types and bias matrices. "
            f"Expected {expected_count} bias matrices for jet types {expected_jet_types}, "
            f"but found {available_count} bias matrices: {available_bias_matrices}. "
            f"Each jet type defined in the event configuration must have a corresponding "
            f"overlap matrix (ak5_overlap_mask, ak8_overlap_mask, ak15_overlap_mask)."
        )
        
        # Validate that specific jet types have corresponding matrices
        for jet_type in expected_jet_types:
            jet_name = jet_type.upper().replace('JETS', '')  # AK5Jets -> AK5
            assert jet_name in available_bias_matrices, (
                f"Missing bias matrix for jet type '{jet_type}'. "
                f"Expected to find '{jet_type_mapping.get(jet_type, f'{jet_name.lower()}_overlap_mask')}' "
                f"in one of the sources."
            )


def validate_overlap_matrix_symmetry(matrix: Tensor, tolerance: float = 1e-6) -> bool:
    """
    Validate that an overlap matrix is symmetric.
    
    Args:
        matrix: Overlap matrix [B, N, N] or [N, N]
        tolerance: Tolerance for symmetry check
        
    Returns:
        True if matrix is symmetric within tolerance
    """
    if matrix.dim() == 2:
        return torch.allclose(matrix, matrix.T, atol=tolerance)
    elif matrix.dim() == 3:
        return torch.allclose(matrix, matrix.transpose(-2, -1), atol=tolerance)
    else:
        raise ValueError(f"Expected 2D or 3D matrix, got {matrix.dim()}D")


def get_jet_type_indices(ak5_size: int = 10, ak8_size: int = 2, ak15_size: int = 2) -> Tuple[slice, slice, slice]:
    """
    Get slice indices for different jet types in the concatenated sequence.
    
    Args:
        ak5_size: Number of AK5 jets (default: 10)
        ak8_size: Number of AK8 jets (default: 2)
        ak15_size: Number of AK15 jets (default: 2)
        
    Returns:
        Tuple of (ak5_slice, ak8_slice, ak15_slice) for indexing the concatenated sequence
    """
    ak5_slice = slice(0, ak5_size)
    ak8_slice = slice(ak5_size, ak5_size + ak8_size)
    ak15_slice = slice(ak5_size + ak8_size, ak5_size + ak8_size + ak15_size)
    
    return ak5_slice, ak8_slice, ak15_slice
