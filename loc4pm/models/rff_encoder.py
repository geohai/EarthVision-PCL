"""Random Fourier feature based location encoder for LOC4PM.

This module implements a simple random Fourier feature (RFF) location encoder
that mirrors the GeoCLIP architecture but adds an optional temporal branch
encoding the day of the year (DOY).  The spatial branch encodes the
normalized longitude/latitude coordinates using a Gaussian RFF and the
temporal branch (if enabled) encodes a normalized day‑of‑year scalar.  The
resulting features are passed through a lightweight multi‑layer perceptron
("capsule") followed by a linear head to produce a fixed‑size embedding.

The RFF encoder is designed to be a drop‑in alternative to the GeoCLIP
location encoder and can be selected via ``loc_name="rff"`` in the model
configuration.  When ``variant="doy"`` the temporal branch is enabled and
expects the ``month`` argument of the ``forward`` method to contain day‑of‑year
indices (1–365).  For other variants or when ``variant`` is ``None``, only
the spatial branch is used.
"""

from __future__ import annotations

from typing import Optional, Sequence

import math
import torch
from torch import nn

from .gaussian_encoding import GaussianEncoding
from .direct import Direct

__all__ = ["RFFLocationEncoder"]


class RFFLocationEncoder(nn.Module):
    """Random Fourier feature based location encoder with optional temporal branch.

    Parameters
    ----------
    emb_dim : int, default=256
        Output embedding dimension.  The final head will map the intermediate
        representation to this size.
    sigma : Sequence[float], default=(2.0, 4.0)
        Standard deviations for the Gaussian kernels used in the spatial RFF.
        Multiple values will produce features at different spatial frequencies.
    encoded_size : int, default=256
        Number of random features to draw *per sigma* for the spatial branch.
    dim : int, default=512
        Hidden dimension inside the capsule network.  The intermediate
        representation after concatenation of the RFF encodings is projected
        through three ``dim->2*dim`` layers before being mapped to ``emb_dim``.
    variant : Optional[str], default=None
        Temporal variant to enable.  If ``"doy"`` (case‑insensitive), the
        encoder will accept a day‑of‑year tensor in the ``month`` argument
        of ``forward`` and apply an additional RFF encoding to it.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        sigma: Sequence[float] = (2.0, 4.0),
        encoded_size: int = 256,
        dim: int = 512,
        variant: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.variant = str(variant).lower() if variant is not None else None
        self.use_temporal = self.variant == "doy"
        # Normalize lon/lat into [-1, 1] range using Direct
        self.pos = Direct(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)
        # Spatial RFF encoder: encodes 2D position into a high dimensional feature
        self.spatial_enc = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=encoded_size)
        # Temporal RFF encoder: encodes day‑of‑year (scalar) if requested
        # Use the same sigma values as the spatial branch for simplicity
        if self.use_temporal:
            self.temporal_enc = GaussianEncoding(sigma=sigma, input_size=1, encoded_size=encoded_size)
        # Feature dimensionalities
        spatial_feat_dim = 2 * encoded_size * len(self.spatial_enc.sigmas)
        temporal_feat_dim = 2 * encoded_size * len(sigma) if self.use_temporal else 0
        total_feat_dim = spatial_feat_dim + temporal_feat_dim
        # Capsule: three linear layers with ReLU, doubling hidden dimension each time
        self.capsule = nn.Sequential(
            nn.Linear(total_feat_dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, 2 * dim),
            nn.ReLU(),
        )
        # Head: project to emb_dim
        self.head = nn.Linear(2 * dim, self.emb_dim)

    def forward(self, coords: torch.Tensor, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode location (and optional day‑of‑year) into a dense embedding.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape ``[B, 2]`` containing latitude and longitude in degrees.
        month : Optional[torch.Tensor], default=None
            Tensor containing temporal indices.  When ``variant"doy"`` was
            specified at construction, this argument is interpreted as the
            day‑of‑year and must therefore lie in the range 1–365.  Otherwise
            it is ignored.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``[B, emb_dim]``.
        """
        if coords is None:
            raise ValueError("coords is required for RFFLocationEncoder")
        if coords.dim() != 2 or coords.size(-1) != 2:
            raise ValueError(
                f"coords must have shape [B, 2], got {tuple(coords.shape)}"
            )
        # Reorder (lat, lon) to (lon, lat) for normalization and encode into [-1,1]
        lonlat = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
        loc = self.pos(lonlat)
        # Encode spatial features
        spatial_feats = self.spatial_enc(loc)
        features = [spatial_feats]
        # Encode temporal features if requested
        if self.use_temporal:
            if time is None:
                raise ValueError(
                    "time (day-of-year) tensor must be provided when variant='doy'"
                )
            # Flatten and convert to float
            t = time.float().squeeze(-1) if time.dim() > 1 else time.float()
            # Normalize day of year to [0,1]; adding small epsilon to avoid zeros at exactly 0
            t_norm = t / 365.2425
            t_norm = t_norm.unsqueeze(-1)  # shape [B,1]
            temporal_feats = self.temporal_enc(t_norm)
            features.append(temporal_feats)
        # Concatenate features and run through capsule and head
        feat = torch.cat(features, dim=-1)
        x = self.capsule(feat)
        out = self.head(x)
        return out