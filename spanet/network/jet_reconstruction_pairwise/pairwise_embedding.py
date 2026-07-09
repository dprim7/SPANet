"""Embedding layers with pairwise interaction support.

Extends the standard multi-input embedding to compute and return a pairwise
attention bias over the FULL concatenated sequence of jet collections.

Design rationale (multi-collection pairwise):
    SPANet's encoder attends over one combined sequence built by concatenating
    every input collection (e.g. Jets + BoostedJets + VeryBoostedJets), so the
    pairwise bias must be an N_total x N_total matrix aligned with that
    sequence. We compute the ParT-style features (ln kt, ln z, ln deltaR,
    ln m^2) for ALL participating collections at once -- covering both
    intra-collection pairs (Jet-Jet, Boosted-Boosted) and cross-collection
    pairs (Jet-Boosted), which matter physically (e.g. a semi-resolved top is
    a small-R b-jet paired with a boosted W fatjet).

    The enabling prerequisite is denormalization: each collection is z-scored
    with its OWN statistics (and pt is log-transformed), so normalized values
    from different collections are not comparable. Every collection is
    therefore denormalized back to physical units (GeV / radians) with its own
    statistics BEFORE the kinematics are concatenated; only then are
    cross-collection deltaR / m^2 meaningful. Kinematics are concatenated in
    the SAME order the embeddings are stacked so that bias position k refers
    to the same jet as attention position k.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from spanet.options import Options
from spanet.dataset.types import InputType
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset

from spanet.network.layers.linear_block import create_linear_block
from spanet.network.layers.embedding.combined_vector_embedding import CombinedVectorEmbedding

from .pairwise_features import (
    PairwiseFeatureComputer,
    PairwiseEmbedding,
    auto_detect_kinematic_features,
    combine_collection_kinematics,
    sincos_to_phi,
)


class MultiInputVectorEmbeddingWithPairwise(nn.Module):
    """Multi-input embedding with multi-collection pairwise feature computation."""

    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super().__init__()

        self.options = options

        self.vector_embedding_layers = nn.ModuleList(
            [
                CombinedVectorEmbedding(options, training_dataset, input_name, input_type)
                for input_name, input_type in training_dataset.event_info.input_types.items()
            ]
        )

        self.final_embedding_layer = create_linear_block(
            options,
            options.position_embedding_dim + options.hidden_dim,
            options.hidden_dim,
            options.skip_connections,
        )

        self._setup_pairwise_features(options, training_dataset)

    def _setup_pairwise_features(self, options: Options, training_dataset: JetReconstructionDataset):
        event_info = training_dataset.event_info

        # Which SEQUENTIAL collections participate. "" or "all" -> every
        # sequential collection (default); a single name or list restricts it.
        requested = getattr(options, "pairwise_input_source", "")
        if isinstance(requested, str):
            requested_set = None if requested.strip().lower() in ("", "all") else {requested}
        else:
            requested_set = set(requested)

        self.pairwise_collections = []
        found_names = []
        for idx, (input_name, input_type) in enumerate(event_info.input_types.items()):
            if input_type != InputType.Sequential:
                continue
            if requested_set is not None and input_name not in requested_set:
                continue

            feature_infos = event_info.input_features[input_name]
            feature_names = [f.name for f in feature_infos]
            kin = auto_detect_kinematic_features(feature_names, input_name)

            info = {
                "input_index": idx,
                "name": input_name,
                "kin": kin,
                "pt_is_log": bool(feature_infos[kin["pt_idx"]].log_scale),
                "pt_is_normalized": bool(feature_infos[kin["pt_idx"]].normalize),
                "eta_is_normalized": bool(feature_infos[kin["eta_idx"]].normalize),
                # sinphi/cosphi are typically not normalized (they're in [-1, 1]).
                "phi_is_normalized": (
                    bool(feature_infos[kin["phi_idx"]].normalize)
                    if (not kin["use_sincos_phi"] and kin["phi_idx"] >= 0)
                    else False
                ),
                "mass_is_normalized": bool(
                    kin["mass_idx"] >= 0 and feature_infos[kin["mass_idx"]].normalize
                ),
            }

            # Per-collection normalization statistics for denormalizing back to
            # physical units. Each collection MUST be denormalized with its own
            # stats before cross-collection features are computed.
            if options.normalize_features:
                if training_dataset.mean is None:
                    training_dataset.compute_source_statistics()
                self.register_buffer(
                    f"_pw_mean_{idx}", training_dataset.mean[input_name].clone().float()
                )
                self.register_buffer(
                    f"_pw_std_{idx}", training_dataset.std[input_name].clone().float()
                )

            self.pairwise_collections.append(info)
            found_names.append(input_name)

        if not self.pairwise_collections:
            raise ValueError(
                "Could not find any SEQUENTIAL input for pairwise features. "
                f"Specified: '{requested}', "
                f"Available: {list(event_info.input_types.keys())}"
            )

        self.pairwise_computer = PairwiseFeatureComputer(num_features=options.num_pairwise_features)
        self.pairwise_embedding = PairwiseEmbedding(
            num_features=options.num_pairwise_features,
            num_heads=options.num_attention_heads,
            embed_dim=options.pairwise_embedding_dim,
        )

    def _extract_kinematics(
        self, info: dict, source_data: Tensor, source_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Recover PHYSICAL (pt, eta, phi, mass) for one collection.

        Uses this collection's own normalization statistics. With
        `log_normalize` the dataset applies log(pt+1) FIRST and then z-scores,
        so the stored value is (log(pt+1) - mean) / std: we must undo the
        normalization BEFORE undoing the log.
        """
        idx = info["kin"]
        mean = getattr(self, f"_pw_mean_{info['input_index']}", None)
        std = getattr(self, f"_pw_std_{info['input_index']}", None)

        pt = source_data[:, :, idx["pt_idx"]]
        if info["pt_is_normalized"] and mean is not None:
            pt = pt * std[idx["pt_idx"]] + mean[idx["pt_idx"]]
        if info["pt_is_log"]:
            # Dataset transform is log(pt + 1). Recover pt.
            pt = torch.expm1(pt).clamp(min=0)

        eta = source_data[:, :, idx["eta_idx"]]
        if info["eta_is_normalized"] and mean is not None:
            eta = eta * std[idx["eta_idx"]] + mean[idx["eta_idx"]]

        if idx["use_sincos_phi"]:
            sinphi = source_data[:, :, idx["sinphi_idx"]]
            cosphi = source_data[:, :, idx["cosphi_idx"]]
            phi = sincos_to_phi(sinphi, cosphi)
        else:
            phi = source_data[:, :, idx["phi_idx"]]
            if info["phi_is_normalized"] and mean is not None:
                phi = phi * std[idx["phi_idx"]] + mean[idx["phi_idx"]]

        mass = None
        if idx["mass_idx"] >= 0:
            mass = source_data[:, :, idx["mass_idx"]]
            if info["mass_is_normalized"] and mean is not None:
                mass = mass * std[idx["mass_idx"]] + mean[idx["mass_idx"]]
                mass = mass.clamp(min=0)

        return pt, eta, phi, mass, source_mask

    def forward(
        self, sources: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        embeddings = []
        padding_masks = []
        sequence_masks = []
        global_masks = []

        for input_index, vector_embedding_layer in enumerate(self.vector_embedding_layers):
            source_data, source_mask = sources[input_index]
            current_embeddings = vector_embedding_layer(source_data, source_mask)

            embeddings.append(current_embeddings[0])
            padding_masks.append(current_embeddings[1])
            sequence_masks.append(current_embeddings[2])
            global_masks.append(current_embeddings[3])

        embeddings = torch.cat(embeddings, dim=0)
        padding_masks = torch.cat(padding_masks, dim=1)
        sequence_masks = torch.cat(sequence_masks, dim=0)
        global_masks = torch.cat(global_masks, dim=0)

        embeddings = self.final_embedding_layer(embeddings, sequence_masks)

        # Assemble physical kinematics for the full combined sequence, in the
        # SAME order the embeddings were concatenated above. Non-participating
        # segments (e.g. global inputs) contribute masked-out positions.
        by_index = {info["input_index"]: info for info in self.pairwise_collections}
        segments = []
        for input_index in range(len(self.vector_embedding_layers)):
            source_data, source_mask = sources[input_index]
            length = source_data.shape[1] if source_data.dim() == 3 else 1
            info = by_index.get(input_index)
            if info is None:
                segments.append((length, None))
            else:
                segments.append((length, self._extract_kinematics(info, source_data, source_mask)))

        pt, eta, phi, mass, mask = combine_collection_kinematics(segments)

        total_seq_len = embeddings.shape[0]
        if pt.shape[1] != total_seq_len:
            raise RuntimeError(
                f"Pairwise kinematic sequence length ({pt.shape[1]}) does not match "
                f"the embedding sequence length ({total_seq_len}); the pairwise bias "
                "would be misaligned with the attention positions."
            )

        pairwise_features, pair_mask = self.pairwise_computer(pt, eta, phi, mass, mask)
        pairwise_bias = self.pairwise_embedding(pairwise_features, pair_mask)

        return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias
