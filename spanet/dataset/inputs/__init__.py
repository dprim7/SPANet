import h5py
import numpy as np

from spanet.dataset.event_info import EventInfo

from spanet.dataset.types import InputType
from spanet.dataset.inputs.BaseInput import BaseInput
from spanet.dataset.inputs.GlobalInput import GlobalInput
from spanet.dataset.inputs.RelativeInput import RelativeInput
from spanet.dataset.inputs.SequentialInput import SequentialInput
from spanet.dataset.inputs.AttentionBiasInput import AttentionBiasInput
from spanet.dataset.inputs.MultiJetAttentionBiasInput import MultiJetAttentionBiasInput


def create_source_input(
        event_info: EventInfo,
        hdf5_file: h5py.File,
        input_name: str,
        num_events: int,
        limit_index: np.ndarray
) -> BaseInput:
    input_type = event_info.input_type(input_name)
    
    # Special handling for attention bias inputs
    if input_type == InputType.AttentionBias:
        # Check if this is a multi-jet attention bias input
        # by looking for jet-type-specific features
        feature_names = [feature.name for feature in event_info.input_features[input_name]]
        
        # If we have multiple jet-type-specific features, use MultiJetAttentionBiasInput
        multi_jet_features = [name for name in feature_names 
                            if any(jet_type in name.lower() 
                                  for jet_type in ['ak5', 'ak8', 'ak15'])]
        
        if len(multi_jet_features) > 1:
            source_class = MultiJetAttentionBiasInput
        else:
            source_class = AttentionBiasInput
    else:
        # Standard routing for other input types
        source_class = {
            InputType.Sequential: SequentialInput,
            InputType.Relative: RelativeInput,
            InputType.Global: GlobalInput,
        }[input_type]

    return source_class(event_info, hdf5_file, input_name, num_events, limit_index)
