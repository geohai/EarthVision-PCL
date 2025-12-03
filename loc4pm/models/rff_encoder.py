from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


__all__ = ["RFFLocationEncoder"]

Tensor = torch.Tensor


def equal_earth_projection(coords: Tensor) -> Tensor:
    """
    Equal Earth projection (Savric et al., 2018).

    Parameters
    ----------
    coords : Tensor
        [..., 2] with latitude and longitude in *radians* (lat, lon).

    Returns
    -------
    Tensor
        [..., 2] with projected (y, x) coordinates.
    """
    if coords.shape[-1] != 2:
        raise ValueError(f"equal_earth_projection expects [...,2], got {coords.shape}")
    lat = coords[..., 0]
    lon = coords[..., 1]

    # Equal Earth constants (same as in the original paper).
    A1 = 1.340264
    A2 = -0.081106
    A3 = 0.000893
    A4 = 0.003796

    sin_theta = (math.sqrt(3.0) / 2.0) * torch.sin(lat)
    theta = torch.asin(torch.clamp(sin_theta, -1.0, 1.0))

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4

    denom = 3.0 * (9.0 * A4 * theta8 + 7.0 * A3 * theta6 + 3.0 * A2 * theta2 + A1)
    x = 2.0 * math.sqrt(3.0) * lon * torch.cos(theta) / denom
    y = A4 * theta * theta8 + A3 * theta * theta6 + A2 * theta * theta2 + A1 * theta

    # Return (lat_like, lon_like) ordering to stay consistent with [lat, lon].
    return torch.stack((y, x), dim=-1)


class GaussianEncoding(nn.Module):
    """
    Random Fourier feature (Gaussian) encoding.

    Mirrors the functional form of rff.functional.gaussian_encoding:

        gamma(v) = [cos(2 pi B v), sin(2 pi B v)]

    where B ~ N(0, sigma^2).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian used to sample B.
    input_size : int
        Dimensionality of input vectors.
    encoded_size : int
        Number of rows of B. Output dim is 2 * encoded_size.
    """

    def __init__(self, sigma: float, input_size: int = 2, encoded_size: int = 256) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.input_size = int(input_size)
        self.encoded_size = int(encoded_size)

        # Projection matrix B: [encoded_size, input_size].
        B = torch.randn(self.encoded_size, self.input_size) * self.sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.encoded_size

    def forward(self, v: Tensor) -> Tensor:
        """
        v : [B, input_size]  ->  [B, 2 * encoded_size]
        """
        if v.dim() != 2 or v.size(-1) != self.input_size:
            raise ValueError(
                f"GaussianEncoding expects [B,{self.input_size}], got {tuple(v.shape)}"
            )
        # [B, encoded_size]
        proj = torch.matmul(v, self.B.t())
        proj = 2.0 * math.pi * proj
        return torch.cat((torch.cos(proj), torch.sin(proj)), dim=-1)


class _RFFBlock(nn.Module):
    """
    Single-scale RFF + MLP block f_i(γ(·, σ_i)), i.e. one term in Eq. (3) of GeoCLIP.
    """

    def __init__(
        self,
        sigma: float,
        input_size: int = 2,
        encoded_size: int = 256,
        dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoding = GaussianEncoding(
            sigma=sigma, input_size=input_size, encoded_size=encoded_size
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.out_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = dim

    def forward(self, v: Tensor) -> Tensor:
        rff = self.encoding(v)
        return self.mlp(rff)


class RFFLocationEncoder(nn.Module):
    """
    GeoCLIP-style location encoder with optional temporal RFF branch.

    Spatial branch
    --------------
    Implements:

        L(G) = sum_i f_i(γ(EEP(G), σ_i))

    with multiple _RFFBlock's (different σ_i) summed element-wise.

    Temporal branch
    ---------------
    Activated when `variant` contains "doy" or "monthly".

    * Input is a scalar index (month or DOY).
    * We convert to angle and then to (sin, cos) before RFF so that the
      temporal representation is explicitly cyclic.
    * Temporal branch is also a sum of _RFFBlock's, with its own σ list
      (defaults to the spatial σ list).

    Fusion mode is encoded in `variant`:

        variant="doy"          -> add  (spatial + temporal)
        variant="doy-hadamard" -> hadamard (spatial * temporal)
        variant="doy-concat"   -> concat + small MLP

    Parameters
    ----------
    emb_dim : int
        Output embedding dimension (what the fusion head will see).
    sigma : float or Sequence[float]
        σ values for spatial RFF hierarchy.
    encoded_size : int
        Encoded size per RFF layer (before trig).
    dim : int
        Hidden dimension for per-scale MLPs.
    variant : Optional[str]
        Temporal encoding variant.  If None or does not contain "doy" /
        "monthly", the encoder is purely spatial.
    temporal_sigma : Optional[float or Sequence[float]]
        σ values for temporal RFF hierarchy.  If None, reuse spatial σ list.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        sigma: Union[float, Sequence[float]] = (2.0, 4.0),
        encoded_size: int = 256,
        dim: int = 512,
        variant: Optional[str] = None,
        temporal_sigma: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.encoded_size = encoded_size
        self.hidden_dim = dim

        # --- Spatial hierarchy (GeoCLIP-style) ---
        if isinstance(sigma, (float, int)):
            sigma_list = [float(sigma)]
        else:
            sigma_list = [float(s) for s in sigma]

        self.spatial_blocks = nn.ModuleList(
            [
                _RFFBlock(s, input_size=2, encoded_size=encoded_size, dim=dim)
                for s in sigma_list
            ]
        )

        # --- Temporal branch configuration ---
        var = (variant or "").lower()
        self.variant = var
        self.has_time = ("doy" in var) or ("monthly" in var)

        # Encoded fusion mode from variant suffix.
        if "had" in var:
            self.time_fusion = "hadamard"
        elif "cat" in var or "concat" in var:
            self.time_fusion = "concat"
        else:
            self.time_fusion = "add"

        if self.has_time:
            if temporal_sigma is None:
                temporal_sigma_list = sigma_list
            elif isinstance(temporal_sigma, (float, int)):
                temporal_sigma_list = [float(temporal_sigma)]
            else:
                temporal_sigma_list = [float(s) for s in temporal_sigma]

            # Temporal input is 2D (sin, cos) of the index.
            self.temporal_blocks = nn.ModuleList(
                [
                    _RFFBlock(s, input_size=2, encoded_size=encoded_size, dim=dim)
                    for s in temporal_sigma_list
                ]
            )
        else:
            self.temporal_blocks = None

        # Fusion MLP for concat mode.
        if self.has_time and self.time_fusion == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.fusion_mlp = None

        # Final projection to emb_dim and LayerNorm for stability.
        self.head = nn.Linear(dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _encode_spatial(self, coords: Tensor) -> Tensor:
        """
        coords : [B, 2] in degrees (lat, lon).
        """
        if coords.dim() != 2 or coords.size(-1) != 2:
            raise ValueError(
                f"RFFLocationEncoder expects coords [B,2], got {tuple(coords.shape)}"
            )

        # Convert to radians and apply Equal Earth projection.
        lat_rad = coords[:, 0] * math.pi / 180.0
        lon_rad = coords[:, 1] * math.pi / 180.0
        proj = equal_earth_projection(torch.stack((lat_rad, lon_rad), dim=-1))

        feats = None
        for block in self.spatial_blocks:
            out = block(proj)
            feats = out if feats is None else feats + out
        return feats

    def _encode_temporal(self, idx: Tensor) -> Tensor:
        """
        idx : [B] or [B,1]; interpreted as month (1..12) or DOY (1..365)
        depending on the variant.
        """
        if not self.has_time or self.temporal_blocks is None:
            raise RuntimeError(
                "Temporal encoding requested but temporal branch is not configured."
            )

        if idx.dim() > 1:
            idx = idx.squeeze(-1)
        idx = idx.to(dtype=torch.float32)

        if "monthly" in self.variant:
            period = 12.0
        else:  # default to DOY
            period = 365.0

        phi = 2.0 * math.pi * (idx / period)
        t2d = torch.stack((torch.sin(phi), torch.cos(phi)), dim=-1)  # [B, 2]

        feats = None
        for block in self.temporal_blocks:
            out = block(t2d)
            feats = out if feats is None else feats + out
        return feats

    # -------------------------
    # Public forward
    # -------------------------

    def forward(self, coords: Tensor, month_or_doy: Optional[Tensor] = None) -> Tensor:
        """
        coords      : [B, 2]  (lat, lon in degrees)
        month_or_doy: [B] or [B,1]; used as:
            - month (1..12) if \"monthly\" in variant
            - day-of-year (1..365) if \"doy\" in variant
        """
        spatial = self._encode_spatial(coords)

        if self.has_time and month_or_doy is not None:
            temporal = self._encode_temporal(month_or_doy)

            if self.time_fusion == "hadamard":
                fused = spatial * temporal
            elif self.time_fusion == "concat":
                fused = self.fusion_mlp(
                    torch.cat((spatial, temporal), dim=-1)
                )  # type: ignore[arg-type]
            else:  # "add"
                fused = spatial + temporal
        else:
            fused = spatial

        out = self.head(fused)
        return self.norm(out)