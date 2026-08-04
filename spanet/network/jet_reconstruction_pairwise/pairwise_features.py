"""Pairwise interaction features for Particle Transformer-style attention.

This module computes pairwise geometric features between particles and embeds
them into attention bias format. The features are added to attention scores
before softmax, implementing: attention = softmax((QK^T)/sqrt(d) + U) @ V

Reference: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""

import math
from typing import Dict, List, Optional

import torch
from torch import nn, Tensor


def sincos_to_phi(sinphi: Tensor, cosphi: Tensor) -> Tensor:
    """Convert sin(phi) and cos(phi) to phi angle using atan2."""
    return torch.atan2(sinphi, cosphi)


def delta_phi(phi1: Tensor, phi2: Tensor) -> Tensor:
    """Compute delta phi with proper wrapping to [-pi, pi).

    torch.remainder (NOT fmod): fmod keeps the dividend's sign, so for
    dphi < -pi the shifted value is negative and comes back unwrapped --
    |delta phi| in (pi, 2pi) on ~1/8 of uniform-phi pairs. remainder follows
    the divisor's sign, mapping into [0, 2pi) before the -pi shift.
    """
    dphi = phi1 - phi2
    dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
    return dphi


def _find_feature(
    names: List[str],
    exact: set,
    suffixes: tuple,
    exclude: set = frozenset(),
) -> int:
    """Find a kinematic feature by exact name first, then by '_'-prefixed suffix."""
    for i, name in enumerate(names):
        if i not in exclude and name in exact:
            return i
    for i, name in enumerate(names):
        if i in exclude:
            continue
        for suffix in suffixes:
            if name.endswith(suffix):
                return i
    return -1


def auto_detect_kinematic_features(
    feature_names: List[str],
    input_source: str = "",
) -> Dict:
    """Auto-detect pt, eta, phi (or sinphi/cosphi), and mass indices.

    Matches either the bare kinematic name (``pt``) or any prefixed variant
    (``fj_pt``, ``vfj_pt``, ...) so every jet collection is detected regardless
    of its feature-name prefix. Softdrop mass (``*_sdmass``) is intentionally
    NOT matched as mass -- only ``mass`` / ``*_mass`` (the full jet mass).
    """
    feature_names_lower = [f.lower() for f in feature_names]

    result = {
        "pt_idx": -1,
        "eta_idx": -1,
        "phi_idx": -1,
        "sinphi_idx": -1,
        "cosphi_idx": -1,
        "mass_idx": -1,
        "use_sincos_phi": False,
    }

    result["pt_idx"] = _find_feature(feature_names_lower, {"pt"}, ("_pt",))
    result["eta_idx"] = _find_feature(feature_names_lower, {"eta"}, ("_eta",))
    result["mass_idx"] = _find_feature(feature_names_lower, {"mass"}, ("_mass",))

    result["sinphi_idx"] = _find_feature(feature_names_lower, {"sinphi", "sin_phi"}, ("_sinphi",))
    result["cosphi_idx"] = _find_feature(feature_names_lower, {"cosphi", "cos_phi"}, ("_cosphi",))
    if result["sinphi_idx"] != -1 and result["cosphi_idx"] != -1:
        result["use_sincos_phi"] = True
    else:
        # Fall back to a direct phi feature; exclude any lone sin/cos hit so a
        # name like `sin_phi` is never mistaken for phi via the `_phi` suffix.
        exclude = {i for i in (result["sinphi_idx"], result["cosphi_idx"]) if i != -1}
        result["sinphi_idx"] = -1
        result["cosphi_idx"] = -1
        result["phi_idx"] = _find_feature(feature_names_lower, {"phi"}, ("_phi",), exclude)

    if result["pt_idx"] == -1:
        raise ValueError(f"Could not find 'pt' feature in {input_source}: {feature_names}")
    if result["eta_idx"] == -1:
        raise ValueError(f"Could not find 'eta' feature in {input_source}: {feature_names}")
    if result["phi_idx"] == -1 and not result["use_sincos_phi"]:
        raise ValueError(
            f"Could not find 'phi' or 'sinphi'/'cosphi' features in {input_source}: {feature_names}"
        )

    return result


# Canonical pairwise feature names, in legacy prefix order (the first four
# are the ParT set; "mink" is the PELICAN-style Lorentz invariant).
CANONICAL_PAIRWISE_FEATURES = ("kt", "z", "dr", "m2", "mink")

_FEATURE_SET_ALIASES = {
    "part4": ("kt", "z", "dr", "m2"),
    "mink1": ("mink",),
}


def parse_pairwise_feature_set(spec: str, legacy_num_features: int):
    """Resolve options into an ordered tuple of pairwise feature names.

    ``spec`` is a "+"- or ","-joined token list ("part4+mink", "mink",
    "m2,mink", ...); aliases expand in place; duplicates are dropped keeping
    first occurrence. An empty spec selects the legacy behavior: the first
    ``legacy_num_features`` of (kt, z, dr, m2) -- values outside 1..4 are
    rejected (the old code silently capped at 4).
    """
    spec = (spec or "").strip().lower()
    if not spec:
        if not 1 <= legacy_num_features <= 4:
            raise ValueError(
                f"num_pairwise_features={legacy_num_features} is invalid without "
                f"pairwise_feature_set; the legacy prefix supports 1..4 "
                f"({CANONICAL_PAIRWISE_FEATURES[:4]}). Set pairwise_feature_set "
                f"(e.g. 'part4+mink') to use other combinations."
            )
        return CANONICAL_PAIRWISE_FEATURES[:legacy_num_features]

    names = []
    for token in spec.replace("+", ",").split(","):
        token = token.strip()
        if not token:
            continue
        expanded = _FEATURE_SET_ALIASES.get(token, (token,))
        for name in expanded:
            if name not in CANONICAL_PAIRWISE_FEATURES:
                raise ValueError(
                    f"Unknown pairwise feature '{name}' in pairwise_feature_set="
                    f"'{spec}'. Known: {CANONICAL_PAIRWISE_FEATURES} "
                    f"(aliases: {tuple(_FEATURE_SET_ALIASES)})."
                )
            if name not in names:
                names.append(name)
    if not names:
        raise ValueError(f"pairwise_feature_set='{spec}' selects no features")
    return tuple(names)


def extract_physical_kinematics(
    source_data: Tensor,
    kin: dict,
    pt_is_log: bool,
    mass_is_log: bool,
):
    """Recover PHYSICAL (pt, eta, phi, mass_or_None) from a raw source tensor.

    The dataset pipeline applies ONLY the log transform at load time
    (``SequentialInput.load``: ``log(x + 1)`` for ``log_scale`` features);
    z-scoring happens later, out-of-place, inside ``CombinedVectorEmbedding``
    via ``Normalizer`` and never touches the raw ``sources`` tensors this
    function receives. Recovery is therefore:

        pt   = expm1(x)  if the pt feature has log_scale, else x
        eta  = x
        phi  = atan2(sinphi, cosphi)  (or raw phi column)
        mass = expm1(x)  if the mass feature has log_scale, else x

    No mean/std enters anywhere -- applying ``x*std + mean`` here (as an
    earlier revision did) un-z-scores data that was never z-scored and
    nonlinearly distorts every downstream pairwise feature.
    """
    pt = source_data[:, :, kin["pt_idx"]]
    if pt_is_log:
        pt = torch.expm1(pt).clamp(min=0)

    eta = source_data[:, :, kin["eta_idx"]]

    if kin["use_sincos_phi"]:
        phi = sincos_to_phi(
            source_data[:, :, kin["sinphi_idx"]],
            source_data[:, :, kin["cosphi_idx"]],
        )
    else:
        phi = source_data[:, :, kin["phi_idx"]]

    mass = None
    if kin["mass_idx"] >= 0:
        mass = source_data[:, :, kin["mass_idx"]]
        if mass_is_log:
            mass = torch.expm1(mass).clamp(min=0)

    return pt, eta, phi, mass


def combine_collection_kinematics(
    segments: List,
) -> "tuple":
    """Assemble per-collection physical kinematics into one combined sequence.

    ``segments`` is an ordered list matching the embedding concatenation order.
    Each element is ``(length, kin)`` where ``kin`` is either ``None`` (a
    segment that does not participate in pairwise features -- its positions
    are masked out) or a tuple ``(pt, eta, phi, mass_or_None, mask)`` of
    ``(B, length)`` tensors already converted to PHYSICAL units.

    Returns ``(pt, eta, phi, mass, mask)`` of shape ``(B, N_total)``.

    Rationale: raw source tensors carry log1p-transformed values for
    ``log_scale`` features (z-scoring happens only downstream inside
    ``CombinedVectorEmbedding`` on separate tensors), so collections are on
    mixed scales until recovered. Cross-collection pairwise features
    (deltaR, m2, kt, z) are only physically meaningful after every collection
    is recovered to a common physical scale (GeV / radians) via
    ``extract_physical_kinematics``. This function only aligns and
    concatenates.
    """
    reference = None
    for _, kin in segments:
        if kin is not None:
            reference = kin[0]
            break
    if reference is None:
        raise ValueError("combine_collection_kinematics: no participating segments")

    batch_size = reference.shape[0]
    total = sum(length for length, _ in segments)
    device, dtype = reference.device, reference.dtype

    pt = torch.zeros(batch_size, total, device=device, dtype=dtype)
    eta = torch.zeros(batch_size, total, device=device, dtype=dtype)
    phi = torch.zeros(batch_size, total, device=device, dtype=dtype)
    mass = torch.zeros(batch_size, total, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, total, device=device, dtype=torch.bool)

    offset = 0
    for length, kin in segments:
        if kin is not None:
            seg_pt, seg_eta, seg_phi, seg_mass, seg_mask = kin
            pt[:, offset:offset + length] = seg_pt
            eta[:, offset:offset + length] = seg_eta
            phi[:, offset:offset + length] = seg_phi
            if seg_mass is not None:
                mass[:, offset:offset + length] = seg_mass
            mask[:, offset:offset + length] = seg_mask
        offset += length

    return pt, eta, phi, mass, mask


class PairwiseFeatureComputer(nn.Module):
    """Compute Particle Transformer-style pairwise interaction features.

    Supports both same-type blocks (a collection with itself: square, diagonal
    removed) and cross-type blocks (two different collections: rectangular),
    via :meth:`compute_block`. The classic single-set :meth:`forward` is kept
    for the unified (shared-MLP) path and existing tests.
    """

    def __init__(self, num_features: int = 4, eps: float = 1e-7, feature_names=None):
        super().__init__()
        # feature_names wins when given; otherwise legacy prefix of the
        # canonical order (kt, z, dr, m2) selected by num_features.
        if feature_names is None:
            feature_names = parse_pairwise_feature_set("", num_features)
        self.feature_names = tuple(feature_names)
        self.num_features = len(self.feature_names)
        self.eps = eps

    @staticmethod
    def _four_vector(pt: Tensor, eta: Tensor, phi: Tensor, mass: Tensor, eps: float):
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt((pt * torch.cosh(eta)) ** 2 + mass**2 + eps)
        return px, py, pz, energy

    def compute_block(
        self,
        kin_i,
        kin_j,
        mask_i: Optional[Tensor] = None,
        mask_j: Optional[Tensor] = None,
        remove_self_pair: bool = False,
    ):
        """Features between set i (rows) and set j (columns).

        ``kin_*`` are ``(pt, eta, phi, mass_or_None)`` tuples of ``(B, N)``
        tensors in PHYSICAL units. Returns ``(features, pair_mask)`` with
        shapes ``(B, Ni, Nj, F)`` / ``(B, Ni, Nj)``; ``pair_mask`` is None if
        both masks are None. Padded pairs (and the diagonal, when
        ``remove_self_pair``) are zeroed in features and False in the mask.
        """
        pt_i, eta_i, phi_i, mass_i = kin_i
        pt_j, eta_j, phi_j, mass_j = kin_j
        if mass_i is None:
            mass_i = torch.zeros_like(pt_i)
        if mass_j is None:
            mass_j = torch.zeros_like(pt_j)

        # Feature menu (rapidity approximated with pseudorapidity):
        #   kt, z, dr, m2 -- the ParT set: ln(kt), ln(z), ln(deltaR), ln(m^2)
        #   mink -- PELICAN-style Lorentz invariant ln(2 p_i.p_j); note
        #           2 p_i.p_j = m2_ij - m_i^2 - m_j^2, the pair mass with the
        #           self-masses removed, in the same numeric band as ln(m^2).
        need_angular = any(f in self.feature_names for f in ("kt", "z", "dr"))
        need_vectors = any(f in self.feature_names for f in ("m2", "mink"))

        if need_vectors:
            px_i, py_i, pz_i, e_i = self._four_vector(pt_i, eta_i, phi_i, mass_i, self.eps)
            px_j, py_j, pz_j, e_j = self._four_vector(pt_j, eta_j, phi_j, mass_j, self.eps)
            px_i, px_j = px_i.unsqueeze(2), px_j.unsqueeze(1)
            py_i, py_j = py_i.unsqueeze(2), py_j.unsqueeze(1)
            pz_i, pz_j = pz_i.unsqueeze(2), pz_j.unsqueeze(1)
            e_i, e_j = e_i.unsqueeze(2), e_j.unsqueeze(1)

        # Broadcast rows (i, dim 2) against columns (j, dim 1) -> (B, Ni, Nj).
        pt_i, pt_j = pt_i.unsqueeze(2), pt_j.unsqueeze(1)
        eta_i, eta_j = eta_i.unsqueeze(2), eta_j.unsqueeze(1)
        phi_i, phi_j = phi_i.unsqueeze(2), phi_j.unsqueeze(1)

        if need_angular:
            d_eta = eta_i - eta_j
            d_phi = delta_phi(phi_i, phi_j)
            delta_r = torch.sqrt(d_eta**2 + d_phi**2 + self.eps)
            pt_min = torch.minimum(pt_i, pt_j)

        features = []
        for name in self.feature_names:
            if name == "kt":
                kt = pt_min * delta_r
                features.append(torch.log(kt.clamp(min=self.eps)))
            elif name == "z":
                z = pt_min / (pt_i + pt_j + self.eps)
                features.append(torch.log(z.clamp(min=self.eps)))
            elif name == "dr":
                features.append(torch.log(delta_r.clamp(min=self.eps)))
            elif name == "m2":
                m2 = (e_i + e_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2
                features.append(torch.log(m2.clamp(min=self.eps)))
            elif name == "mink":
                dot = e_i * e_j - px_i * px_j - py_i * py_j - pz_i * pz_j
                features.append(torch.log((2.0 * dot).clamp(min=self.eps)))
            else:  # pragma: no cover -- guarded by parse_pairwise_feature_set
                raise ValueError(f"Unknown pairwise feature '{name}'")

        pairwise_features = torch.stack(features, dim=-1)

        square = pairwise_features.shape[1] == pairwise_features.shape[2]
        diag = None
        if remove_self_pair and square:
            # Neutralize self-pairs (i == j), like ParT's remove_self_pair.
            # Avoids large negative logs on the diagonal from delta_r ~ sqrt(eps).
            diag = torch.arange(pairwise_features.shape[1], device=pairwise_features.device)
            pairwise_features = pairwise_features.clone()
            pairwise_features[:, diag, diag, :] = 0.0

        # Valid-pair mask (True = real pair) so downstream normalization can
        # ignore padded pairs; the removed diagonal is excluded too.
        pair_mask = None
        if mask_i is not None and mask_j is not None:
            pair_mask = mask_i.unsqueeze(2) & mask_j.unsqueeze(1)
            pair_mask = pair_mask.clone()
            if diag is not None:
                pair_mask[:, diag, diag] = False
            pairwise_features = pairwise_features * pair_mask.unsqueeze(-1)

        return pairwise_features, pair_mask

    def forward(
        self,
        pt: Tensor,
        eta: Tensor,
        phi: Tensor,
        mass: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Single-set (unified) path: all-vs-all with the diagonal removed."""
        return self.compute_block(
            (pt, eta, phi, mass), (pt, eta, phi, mass), mask, mask, remove_self_pair=True
        )


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d that computes statistics over valid (unmasked) positions only.

    A standard BatchNorm1d over the flattened (B, C, N*N) pairwise tensor would
    include padded pairs (which are zeroed), badly skewing the running mean/var --
    for a low-multiplicity event most of the N*N matrix is padding. This variant
    restricts the batch statistics to valid pairs, matching ParT's behaviour.
    Falls back to standard behaviour when no mask is given.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, L); mask: (B, 1, L) with 1.0 = valid, or None (all valid).
        if self.training:
            if mask is None:
                mean = x.mean(dim=(0, 2))
                var = x.var(dim=(0, 2), unbiased=False)
            else:
                n = mask.sum().clamp(min=1.0)
                mean = (x * mask).sum(dim=(0, 2)) / n
                centered = (x - mean[None, :, None]) * mask
                var = (centered * centered).sum(dim=(0, 2)) / n
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class PairwiseEmbedding(nn.Module):
    """Embed pairwise features into attention bias format (ParT-style).

    ParT uses BatchNorm + 1x1 Conv stacks to embed pairwise features into a
    per-head attention bias. We implement the same idea here, but with
    mask-aware BatchNorm so padded pairs do not corrupt the statistics.
    """

    def __init__(self, num_features: int, num_heads: int, embed_dim: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Work on flattened pairs: (B, F, N*N). Masked BN over feature channels.
        self.input_bn = MaskedBatchNorm1d(num_features)
        self.conv1 = nn.Conv1d(num_features, embed_dim, kernel_size=1)
        self.bn1 = MaskedBatchNorm1d(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.bn2 = MaskedBatchNorm1d(embed_dim)
        self.conv3 = nn.Conv1d(embed_dim, num_heads, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, pairwise_features: Tensor, pair_mask: Optional[Tensor] = None) -> Tensor:
        # Blocks may be rectangular (rows != cols) for cross-type collections.
        batch_size, num_rows, num_cols, num_features = pairwise_features.shape

        # (B, R, C, F) -> (B, F, R*C)
        x = pairwise_features.permute(0, 3, 1, 2).contiguous().view(batch_size, num_features, num_rows * num_cols)

        mask = None
        if pair_mask is not None:
            # (B, R, C) -> (B, 1, R*C) float, 1.0 = valid pair
            mask = pair_mask.reshape(batch_size, 1, num_rows * num_cols).to(x.dtype)

        x = self.input_bn(x, mask)
        x = self.act(self.bn1(self.conv1(x), mask))
        x = self.act(self.bn2(self.conv2(x), mask))
        x = self.conv3(x)  # (B, H, R*C)

        x = x.view(batch_size, self.num_heads, num_rows, num_cols)
        if pair_mask is not None:
            # Zero the bias on invalid pairs (padding, self-pairs, and any
            # positions outside the participating collections). The conv stack
            # has bias terms, so without this the invalid entries would carry a
            # nonzero constant into the attention scores.
            x = x * pair_mask.unsqueeze(1).to(x.dtype)
        x = x.reshape(batch_size * self.num_heads, num_rows, num_cols)
        return x
