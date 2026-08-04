"""Embedding layers with pairwise interaction support.

Extends the standard multi-input embedding to compute and return a pairwise
attention bias over the FULL concatenated sequence of jet collections.

Design rationale (multi-collection pairwise):
    SPANet's encoder attends over one combined sequence built by concatenating
    every input collection (e.g. Jets + BoostedJets + VeryBoostedJets), so the
    pairwise bias must be an N_total x N_total matrix aligned with that
    sequence. We compute the ParT-style features (ln kt, ln z, ln deltaR,
    ln m^2) for ALL participating collections -- covering both intra-collection
    pairs (Jet-Jet, Boosted-Boosted) and cross-collection pairs (Jet-Boosted),
    which matter physically (e.g. a semi-resolved top is a small-R b-jet paired
    with a boosted W fatjet).

    The enabling prerequisite is recovery to physical units: the raw
    ``sources`` tensors carry log1p-transformed values for ``log_scale``
    features (``SequentialInput.load`` applies ``log(x+1)`` at load; z-scoring
    happens ONLY later, out-of-place, inside ``CombinedVectorEmbedding`` via
    ``Normalizer`` and never mutates ``sources``). Each collection is
    therefore recovered to physical units (GeV / radians) by undoing exactly
    the log transform -- no mean/std is involved -- BEFORE pair features are
    computed; only then are cross-collection deltaR / m^2 meaningful.
    Kinematics are laid out in the SAME order the embeddings are stacked so
    that bias position k refers to the same jet as attention position k
    (guarded at runtime).

Two embedding structures are available (options.pairwise_block_embeddings):
    * unified (False): one shared MLP + masked BatchNorm over ALL pairs.
    * block (True): an independent MLP (with its own masked BatchNorm) per
      pair-type block -- one per collection (same-type) and one per
      resonance-coupled collection pair (cross-type). Different pair
      populations (Jet-Jet vs Boosted-Boosted vs Jet-Boosted) have very
      different kinematic distributions; per-block normalization treats each
      population on its own terms instead of pooling them.
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
    extract_physical_kinematics,
    parse_pairwise_feature_set,
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

    @staticmethod
    def _cross_key(name_a: str, name_b: str) -> str:
        """Order-independent module key for a cross-type block."""
        return "__x__".join(sorted((name_a, name_b)))

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
                # The raw sources tensors are log1p-transformed at load for
                # log_scale features and NEVER z-scored (Normalizer runs
                # out-of-place downstream) -- recovery needs only these flags.
                "pt_is_log": bool(feature_infos[kin["pt_idx"]].log_scale),
                "mass_is_log": bool(
                    kin["mass_idx"] >= 0 and feature_infos[kin["mass_idx"]].log_scale
                ),
            }

            self.pairwise_collections.append(info)

        if not self.pairwise_collections:
            raise ValueError(
                "Could not find any SEQUENTIAL input for pairwise features. "
                f"Specified: '{requested}', "
                f"Available: {list(event_info.input_types.keys())}"
            )

        feature_names = parse_pairwise_feature_set(
            getattr(options, "pairwise_feature_set", ""),
            options.num_pairwise_features,
        )
        num_features = len(feature_names)
        num_heads = options.num_attention_heads
        embed_dim = options.pairwise_embedding_dim

        self.pairwise_computer = PairwiseFeatureComputer(feature_names=feature_names)

        self.block_mode = bool(getattr(options, "pairwise_block_embeddings", False))
        if self.block_mode:
            # One independent MLP (with its own masked BatchNorm) per collection.
            self.same_type_embeddings = nn.ModuleDict({
                info["name"]: PairwiseEmbedding(num_features, num_heads, embed_dim)
                for info in self.pairwise_collections
            })

            # Cross-type blocks only for collection pairs coupled by a
            # resonance in the event file (an event particle whose daughters
            # are drawn from more than one participating collection).
            self.cross_pairs: List[Tuple[int, int]] = []
            if getattr(options, "pairwise_cross_type", True):
                seq_set = {info["input_index"] for info in self.pairwise_collections}
                coupled = set()
                for _particle, products in event_info.product_particles.items():
                    sources = sorted({s for s in products.sources if s in seq_set})
                    for a_pos in range(len(sources)):
                        for b_pos in range(a_pos + 1, len(sources)):
                            coupled.add((sources[a_pos], sources[b_pos]))
                self.cross_pairs = sorted(coupled)

            name_of = {info["input_index"]: info["name"] for info in self.pairwise_collections}
            self.cross_type_embeddings = nn.ModuleDict({
                self._cross_key(name_of[a], name_of[b]): PairwiseEmbedding(num_features, num_heads, embed_dim)
                for (a, b) in self.cross_pairs
            })
        else:
            # Unified: one shared MLP + masked BatchNorm over all pairs.
            self.pairwise_embedding = PairwiseEmbedding(
                num_features=num_features,
                num_heads=num_heads,
                embed_dim=embed_dim,
            )

    def _extract_kinematics(
        self, info: dict, source_data: Tensor, source_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
        """Recover PHYSICAL (pt, eta, phi, mass) for one collection.

        Thin wrapper over :func:`extract_physical_kinematics` -- the raw
        ``sources`` tensors are log1p-transformed only (never z-scored), so
        recovery is expm1 for log_scale features and identity otherwise.
        """
        pt, eta, phi, mass = extract_physical_kinematics(
            source_data, info["kin"], info["pt_is_log"], info["mass_is_log"]
        )
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

        total_seq_len = embeddings.shape[0]
        batch_size = embeddings.shape[1]

        # Per-input segment lengths + offsets, in the SAME order the
        # embeddings were concatenated above (alignment is guarded below).
        lengths = []
        for input_index in range(len(self.vector_embedding_layers)):
            source_data, _ = sources[input_index]
            lengths.append(source_data.shape[1] if source_data.dim() == 3 else 1)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        if offsets[-1] != total_seq_len:
            raise RuntimeError(
                f"Pairwise kinematic sequence length ({offsets[-1]}) does not match "
                f"the embedding sequence length ({total_seq_len}); the pairwise bias "
                "would be misaligned with the attention positions."
            )

        by_index = {info["input_index"]: info for info in self.pairwise_collections}

        if self.block_mode:
            # ---- Block mode: per-pair-type MLPs, assembled at offsets. ----
            kinematics = {}
            for idx, info in by_index.items():
                source_data, source_mask = sources[idx]
                kinematics[idx] = self._extract_kinematics(info, source_data, source_mask)

            pairwise_bias = embeddings.new_zeros(
                batch_size * self.options.num_attention_heads, total_seq_len, total_seq_len
            )

            # Same-type blocks: each collection with itself.
            for idx, info in by_index.items():
                pt, eta, phi, mass, mask = kinematics[idx]
                features, pair_mask = self.pairwise_computer.compute_block(
                    (pt, eta, phi, mass), (pt, eta, phi, mass),
                    mask, mask, remove_self_pair=True,
                )
                block_bias = self.same_type_embeddings[info["name"]](features, pair_mask)
                o, L = offsets[idx], lengths[idx]
                pairwise_bias[:, o:o + L, o:o + L] = block_bias

            # Cross-type blocks: resonance-coupled collection pairs.
            for (a, b) in self.cross_pairs:
                pt_a, eta_a, phi_a, mass_a, mask_a = kinematics[a]
                pt_b, eta_b, phi_b, mass_b, mask_b = kinematics[b]
                features, pair_mask = self.pairwise_computer.compute_block(
                    (pt_a, eta_a, phi_a, mass_a), (pt_b, eta_b, phi_b, mass_b),
                    mask_a, mask_b, remove_self_pair=False,
                )
                key = self._cross_key(by_index[a]["name"], by_index[b]["name"])
                block_bias = self.cross_type_embeddings[key](features, pair_mask)

                oa, ob = offsets[a], offsets[b]
                la, lb = lengths[a], lengths[b]
                pairwise_bias[:, oa:oa + la, ob:ob + lb] = block_bias
                # Features are symmetric under i<->j, so the (b, a) block is
                # the transpose of the (a, b) block.
                pairwise_bias[:, ob:ob + lb, oa:oa + la] = block_bias.transpose(1, 2)

            return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias

        # ---- Unified mode: one shared MLP over the combined sequence. ----
        segments = []
        for input_index in range(len(self.vector_embedding_layers)):
            source_data, source_mask = sources[input_index]
            info = by_index.get(input_index)
            if info is None:
                segments.append((lengths[input_index], None))
            else:
                segments.append((lengths[input_index], self._extract_kinematics(info, source_data, source_mask)))

        pt, eta, phi, mass, mask = combine_collection_kinematics(segments)
        pairwise_features, pair_mask = self.pairwise_computer(pt, eta, phi, mass, mask)
        pairwise_bias = self.pairwise_embedding(pairwise_features, pair_mask)

        return embeddings, padding_masks, sequence_masks, global_masks, pairwise_bias
