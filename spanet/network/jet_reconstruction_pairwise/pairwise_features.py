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
    """Compute delta phi with proper wrapping to [-pi, pi]."""
    dphi = phi1 - phi2
    dphi = torch.fmod(dphi + math.pi, 2 * math.pi) - math.pi
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

    Rationale: each collection is normalized with its OWN statistics (and pt is
    log-transformed), so a normalized value from `Jets` and one from
    `BoostedJets` are not comparable. Cross-collection pairwise features
    (deltaR, m2, kt, z) are only physically meaningful after every collection
    has been denormalized back to a common physical scale (GeV / radians).
    Callers must therefore denormalize BEFORE combining -- this function only
    aligns and concatenates.
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
    """Compute Particle Transformer-style pairwise interaction features."""

    def __init__(self, num_features: int = 4, eps: float = 1e-7):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(
        self,
        pt: Tensor,
        eta: Tensor,
        phi: Tensor,
        mass: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_particles = pt.shape

        if mass is None:
            mass = torch.zeros_like(pt)

        # Following ParT: use ln(kt), ln(z), ln(delta), ln(m^2) constructed from kinematics.
        # Here we approximate rapidity with pseudorapidity (eta), which is reasonable for ultra-relativistic jets.
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        energy = torch.sqrt((pt * torch.cosh(eta)) ** 2 + mass**2 + self.eps)

        pt_i, pt_j = pt.unsqueeze(2), pt.unsqueeze(1)
        eta_i, eta_j = eta.unsqueeze(2), eta.unsqueeze(1)
        phi_i, phi_j = phi.unsqueeze(2), phi.unsqueeze(1)
        px_i, px_j = px.unsqueeze(2), px.unsqueeze(1)
        py_i, py_j = py.unsqueeze(2), py.unsqueeze(1)
        pz_i, pz_j = pz.unsqueeze(2), pz.unsqueeze(1)
        e_i, e_j = energy.unsqueeze(2), energy.unsqueeze(1)

        features = []

        d_eta = eta_i - eta_j
        d_phi = delta_phi(phi_i, phi_j)
        delta_r = torch.sqrt(d_eta**2 + d_phi**2 + self.eps)
        pt_min = torch.minimum(pt_i, pt_j)

        kt = pt_min * delta_r
        features.append(torch.log(kt.clamp(min=self.eps)))

        if self.num_features >= 2:
            z = pt_min / (pt_i + pt_j + self.eps)
            features.append(torch.log(z.clamp(min=self.eps)))

        if self.num_features >= 3:
            features.append(torch.log(delta_r.clamp(min=self.eps)))

        if self.num_features >= 4:
            m2 = (e_i + e_j) ** 2 - (px_i + px_j) ** 2 - (py_i + py_j) ** 2 - (pz_i + pz_j) ** 2
            m2 = m2.clamp(min=self.eps)
            features.append(torch.log(m2))

        pairwise_features = torch.stack(features, dim=-1)

        # Neutralize self-pairs (i == j), like ParT's optional remove_self_pair.
        # This avoids injecting large negative logs on the diagonal from delta_r ~ sqrt(eps).
        i = torch.arange(num_particles, device=pt.device)
        pairwise_features[:, i, i, :] = 0.0

        # Valid-pair mask (True = real pair) so downstream normalization can ignore
        # padded pairs. Self-pairs are excluded too (their features are zeroed above).
        pair_mask = None
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)   # (B, N, N)
            pair_mask = pair_mask.clone()
            pair_mask[:, i, i] = False
            pairwise_features = pairwise_features * pair_mask.unsqueeze(-1)

        return pairwise_features, pair_mask


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
        batch_size, seq_len, _, num_features = pairwise_features.shape

        # (B, N, N, F) -> (B, F, N*N)
        x = pairwise_features.permute(0, 3, 1, 2).contiguous().view(batch_size, num_features, seq_len * seq_len)

        mask = None
        if pair_mask is not None:
            # (B, N, N) -> (B, 1, N*N) float, 1.0 = valid pair
            mask = pair_mask.reshape(batch_size, 1, seq_len * seq_len).to(x.dtype)

        x = self.input_bn(x, mask)
        x = self.act(self.bn1(self.conv1(x), mask))
        x = self.act(self.bn2(self.conv2(x), mask))
        x = self.conv3(x)  # (B, H, N*N)

        x = x.view(batch_size, self.num_heads, seq_len, seq_len)
        if pair_mask is not None:
            # Zero the bias on invalid pairs (padding, self-pairs, and any
            # positions outside the participating collections). The conv stack
            # has bias terms, so without this the invalid entries would carry a
            # nonzero constant into the attention scores.
            x = x * pair_mask.unsqueeze(1).to(x.dtype)
        x = x.reshape(batch_size * self.num_heads, seq_len, seq_len)
        return x
