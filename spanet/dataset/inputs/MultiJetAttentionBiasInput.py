"""
Input class for loading multiple jet-type attention bias matrices from HDF5 datasets.

This extends the basic AttentionBiasInput to handle separate overlap matrices for 
different jet types (AK5, AK8, AK15) and combines them into a block-diagonal structure.
"""

import h5py
import numpy as np
import torch
from typing import Optional

from spanet.dataset.inputs.BaseInput import BaseInput
from spanet.dataset.types import SpecialKey, Source, Statistics
from spanet.network.utilities.multi_jet_attention_bias import concatenate_jet_overlap_matrices


class MultiJetAttentionBiasInput(BaseInput):
    """
    Input class for loading attention bias matrices from multiple jet types.
    
    This class handles loading separate overlap matrices for AK5, AK8, and AK15 jets
    and combines them into a single block-diagonal attention bias matrix that matches
    the concatenated jet sequence used by the network.
    """

    def __init__(self, *args, **kwargs):
        # Set default jet sizes before calling super().__init__()
        # This ensures they exist when load() is called
        self.ak5_bias_data = None
        self.ak8_bias_data = None
        self.ak15_bias_data = None
        self.combined_bias_data = None
        self.combined_pairwise_mask = None
        
        # Default jet counts - can be overridden by event info configuration
        self.ak5_size = 10
        self.ak8_size = 2
        self.ak15_size = 2
        
        super().__init__(*args, **kwargs)

    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        """Load multiple jet-type overlap matrices from HDF5."""
        input_group = [SpecialKey.Inputs, self.input_name]

        # Try to load each jet type's overlap matrix
        ak5_data, ak5_mask = self._load_jet_type_data(hdf5_file, input_group, "ak5_overlap", limit_index)
        ak8_data, ak8_mask = self._load_jet_type_data(hdf5_file, input_group, "ak8_overlap", limit_index)
        ak15_data, ak15_mask = self._load_jet_type_data(hdf5_file, input_group, "ak15_overlap", limit_index)
        
        # Store individual matrices (for potential individual access)
        self.ak5_bias_data = ak5_data
        self.ak8_bias_data = ak8_data
        self.ak15_bias_data = ak15_data
        
        # Combine into block-diagonal structure
        # Infer sizes from actual data rather than using defaults
        ak5_actual_size = ak5_data.shape[1] if ak5_data is not None else 0
        ak8_actual_size = ak8_data.shape[1] if ak8_data is not None else 0
        ak15_actual_size = ak15_data.shape[1] if ak15_data is not None else 0
        
        self.combined_bias_data = concatenate_jet_overlap_matrices(
            ak5_data, ak8_data, ak15_data, 
            ak5_actual_size, ak8_actual_size, ak15_actual_size
        )
        
        # Create combined mask for the block-diagonal structure  
        ak5_actual_size = ak5_data.shape[1] if ak5_data is not None else 0
        ak8_actual_size = ak8_data.shape[1] if ak8_data is not None else 0
        ak15_actual_size = ak15_data.shape[1] if ak15_data is not None else 0
        
        self._create_combined_mask(ak5_mask, ak8_mask, ak15_mask, 
                                 ak5_actual_size, ak8_actual_size, ak15_actual_size)

    def _load_jet_type_data(self, hdf5_file: h5py.File, input_group, dataset_name: str, limit_index: np.ndarray):
        """Load overlap matrix for a specific jet type."""
        try:
            # Load bias values: [NUM_EVENTS, MAX_JETS_TYPE, MAX_JETS_TYPE]
            bias_dataset = self.dataset(hdf5_file, input_group, dataset_name)
            bias_data = torch.from_numpy(bias_dataset[:]).contiguous()
            
            # Try to load corresponding mask
            try:
                mask_dataset_name = f"{dataset_name}_mask"
                source_mask = torch.from_numpy(
                    self.dataset(hdf5_file, input_group, mask_dataset_name)[:]
                ).contiguous()
                
                # If mask is 2D, it's a jet mask - create pairwise mask
                if source_mask.dim() == 2:
                    pairwise_mask = source_mask[:, :, None] & source_mask[:, None, :]
                else:
                    pairwise_mask = source_mask
                    
            except KeyError:
                # If no specific mask, assume all entries are valid
                num_events, max_jets, _ = bias_data.shape
                pairwise_mask = torch.ones(num_events, max_jets, max_jets, dtype=torch.bool)
            
            # Mask invalid entries to zero
            bias_data = bias_data * pairwise_mask.float()
            
            # Apply limit index
            bias_data = bias_data[limit_index].contiguous()
            pairwise_mask = pairwise_mask[limit_index].contiguous()
            
            return bias_data, pairwise_mask
            
        except KeyError:
            # Dataset not found for this jet type
            return None, None

    def _create_combined_mask(self, ak5_mask, ak8_mask, ak15_mask, ak5_size, ak8_size, ak15_size):
        """Create combined pairwise mask for the block-diagonal structure."""
        if self.combined_bias_data is None:
            self.combined_pairwise_mask = None
            return
            
        batch_size, total_size, _ = self.combined_bias_data.shape
        device = self.combined_bias_data.device
        dtype = torch.bool
        
        # Initialize combined mask
        combined_mask = torch.zeros(batch_size, total_size, total_size, dtype=dtype, device=device)
        
        # Fill in the block-diagonal mask structure
        current_offset = 0
        
        # AK5 block
        if ak5_mask is not None:
            end_offset = current_offset + ak5_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = ak5_mask
        else:
            # Default to all valid if no specific mask
            end_offset = current_offset + ak5_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = True
        current_offset += ak5_size
        
        # AK8 block
        if ak8_mask is not None:
            end_offset = current_offset + ak8_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = ak8_mask
        else:
            end_offset = current_offset + ak8_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = True
        current_offset += ak8_size
        
        # AK15 block
        if ak15_mask is not None:
            end_offset = current_offset + ak15_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = ak15_mask
        else:
            end_offset = current_offset + ak15_size
            combined_mask[:, current_offset:end_offset, current_offset:end_offset] = True
        
        self.combined_pairwise_mask = combined_mask

    @property
    def reconstructable(self) -> bool:
        """Attention biases are not reconstruction targets."""
        return False

    def limit(self, event_mask):
        """Apply event-level filtering."""
        if self.combined_bias_data is not None:
            self.combined_bias_data = self.combined_bias_data[event_mask].contiguous()
        if self.combined_pairwise_mask is not None:
            self.combined_pairwise_mask = self.combined_pairwise_mask[event_mask].contiguous()
        
        # Also limit individual matrices if they exist
        if self.ak5_bias_data is not None:
            self.ak5_bias_data = self.ak5_bias_data[event_mask].contiguous()
        if self.ak8_bias_data is not None:
            self.ak8_bias_data = self.ak8_bias_data[event_mask].contiguous()
        if self.ak15_bias_data is not None:
            self.ak15_bias_data = self.ak15_bias_data[event_mask].contiguous()

    def compute_statistics(self) -> Statistics:
        """Compute normalization statistics for attention biases."""
        if self.combined_bias_data is None:
            # Return default statistics if no data
            return Statistics(torch.tensor([0.0]), torch.tensor([1.0]))
            
        # Use combined matrix for statistics
        mask = self.combined_pairwise_mask.float()
        masked_data = self.combined_bias_data * mask
        
        # Compute mean and std excluding masked entries
        num_valid = mask.sum()
        if num_valid > 0:
            masked_mean = masked_data.sum() / num_valid
            masked_var = ((masked_data - masked_mean * mask) ** 2 * mask).sum() / num_valid
            masked_std = torch.sqrt(masked_var + 1e-8)
        else:
            masked_mean = torch.tensor(0.0, device=self.combined_bias_data.device)
            masked_std = torch.tensor(1.0, device=self.combined_bias_data.device)
        
        return Statistics(masked_mean.unsqueeze(0), masked_std.unsqueeze(0))

    def num_vectors(self) -> int:
        """Return number of valid vector pairs per event."""
        if self.combined_pairwise_mask is not None:
            return self.combined_pairwise_mask.sum(dim=(1, 2))
        return 0

    def max_vectors(self) -> int:
        """Return maximum number of jets (total across all types)."""
        if self.combined_bias_data is not None:
            return self.combined_bias_data.shape[1]
        return self.ak5_size + self.ak8_size + self.ak15_size

    def __getitem__(self, item) -> Source:
        """Return a Source with combined bias data and individual jet-type biases."""
        if self.combined_bias_data is None:
            # Return empty source if no data loaded
            return Source(
                data=torch.zeros(1, 1),  # Dummy data
                mask=torch.zeros(1, 1, dtype=torch.bool),  # Dummy mask
                ak_overlap_mask=None,
                ak5_overlap_mask=None,
                ak8_overlap_mask=None,
                ak15_overlap_mask=None
            )
        
        # Extract data for this item
        combined_bias = self.combined_bias_data[item]
        combined_mask = self.combined_pairwise_mask[item] if self.combined_pairwise_mask is not None else None
        
        # Extract individual jet type biases for this item
        ak5_bias = self.ak5_bias_data[item] if self.ak5_bias_data is not None else None
        ak8_bias = self.ak8_bias_data[item] if self.ak8_bias_data is not None else None
        ak15_bias = self.ak15_bias_data[item] if self.ak15_bias_data is not None else None
        
        return Source(
            data=combined_bias,  # Combined matrix as main data
            mask=combined_mask,  # Combined validity mask
            ak_overlap_mask=combined_bias,  # For backward compatibility
            ak5_overlap_mask=ak5_bias,  # Individual AK5 matrix
            ak8_overlap_mask=ak8_bias,  # Individual AK8 matrix
            ak15_overlap_mask=ak15_bias  # Individual AK15 matrix
        )
