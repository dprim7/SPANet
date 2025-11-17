import h5py
import numpy as np

import torch

from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput


class AttentionBiasInput(BaseInput):
    """
    Input class for loading attention bias matrices from HDF5 datasets.
    
    Attention biases are continuous values that get added to pre-softmax attention scores
    to model things like AK jet overlap. They have shape [NUM_EVENTS, MAX_JETS, MAX_JETS].
    """

    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Load the main bias feature (should be the first/only feature in the input)
        bias_feature_name = self.event_info.input_features[self.input_name][0].name
        
        # Load bias values: [NUM_EVENTS, MAX_JETS, MAX_JETS]
        bias_dataset = self.dataset(hdf5_file, input_group, bias_feature_name)
        bias_data = torch.from_numpy(bias_dataset[:]).contiguous()
        
        # Try to load a mask - this could be jet-level mask or explicit bias mask
        try:
            source_mask = torch.from_numpy(
                self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:]
            ).contiguous()
            
            # If mask is 2D, it's a jet mask - create pairwise mask
            if source_mask.dim() == 2:
                # Create pairwise mask: [NUM_EVENTS, MAX_JETS, MAX_JETS]
                pairwise_mask = source_mask[:, :, None] & source_mask[:, None, :]
            else:
                # Assume it's already a pairwise mask
                pairwise_mask = source_mask
                
        except KeyError:
            # If no mask, assume all entries are valid
            num_events, max_jets, _ = bias_data.shape
            pairwise_mask = torch.ones(num_events, max_jets, max_jets, dtype=torch.bool)
        
        # Mask invalid entries to zero
        bias_data = bias_data * pairwise_mask.float()
        
        # Apply limit index and store
        self.bias_data = bias_data[limit_index].contiguous()
        self.pairwise_mask = pairwise_mask[limit_index].contiguous()

    @property
    def reconstructable(self) -> bool:
        """Attention biases are not reconstruction targets."""
        return False

    def limit(self, event_mask):
        """Apply event-level filtering."""
        self.bias_data = self.bias_data[event_mask].contiguous()
        self.pairwise_mask = self.pairwise_mask[event_mask].contiguous()

    def compute_statistics(self) -> Statistics:
        """Compute normalization statistics for attention biases."""
        # Only compute stats over valid (masked) entries
        masked_data = self.bias_data[self.pairwise_mask]
        
        if len(masked_data) > 0:
            masked_mean = masked_data.mean()
            masked_std = masked_data.std()
            
            # Avoid division by zero
            if masked_std < 1e-5:
                masked_std = torch.tensor(1.0)
        else:
            # No valid data - use defaults
            masked_mean = torch.tensor(0.0)
            masked_std = torch.tensor(1.0)
        
        # For bias inputs, check if normalization is requested
        normalize_features = self.event_info.normalized_features(self.input_name)
        if not normalize_features[0]:  # First (and likely only) feature
            masked_mean = torch.tensor(0.0)
            masked_std = torch.tensor(1.0)
        
        # Return as single values (will be broadcast as needed)
        return Statistics(masked_mean.unsqueeze(0), masked_std.unsqueeze(0))

    def num_vectors(self) -> int:
        """Return number of valid vector pairs per event."""
        return self.pairwise_mask.sum(dim=(1, 2))

    def max_vectors(self) -> int:
        """Return maximum number of jets (assumes square matrix)."""
        return self.bias_data.shape[1]

    def __getitem__(self, item) -> Source:
        """Return a Source with bias data in the ak_overlap_mask field."""
        bias_values = self.bias_data[item]  # [MAX_JETS, MAX_JETS]
        bias_mask = self.pairwise_mask[item]  # [MAX_JETS, MAX_JETS]
        
        return Source(
            data=bias_values,  # Not used for attention bias, but required by Source
            mask=bias_mask,    # Pairwise validity mask
            ak_overlap_mask=bias_values  # This is what gets passed to attention layers
        )
