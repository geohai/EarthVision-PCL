# loc4pm/models/rff_encoder.py

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .gaussian_encoding import GaussianEncoding  # your RFF impl :contentReference[oaicite:2]{index=2}


class TemporalRFFEncoder(nn.Module):
    """
    Simple temporal encoder: time index -> sin/cos -> RFF -> MLP -> emb_dim.
    Used ONLY for time; spatial still uses rshf GeoCLIP.
    """

    def __init__(
        self,
        emb_dim: int,
        sigma: Union[float, Sequence[float]] = (2.0, 4.0),
        encoded_size: int = 64,
        hidden_dim: int = 256,
        time_variant: str = "doy",
    ) -> None:
        super().__init__()
        self.time_variant = time_variant.lower()

        # 2-D input: sin(phi), cos(phi)
        self.rff = GaussianEncoding(
            sigma=sigma,
            input_size=2,
            encoded_size=encoded_size,
        )
        rff_out_dim = 2 * encoded_size * (len(sigma) if isinstance(sigma, (list, tuple)) else 1)

        self.mlp = nn.Sequential(
            nn.Linear(rff_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: [B] or [B,1], interpreted as:
          - day-of-year if 'doy' in time_variant
          - month       if 'month' in time_variant
        """
        if idx.dim() > 1:
            idx = idx.squeeze(-1)
        idx = idx.to(dtype=torch.float32)

        period = 365.2425 if "doy" in self.time_variant else 12.0
        phi = 2.0 * math.pi * (idx / period)  # [B]
        t2d = torch.stack((torch.sin(phi), torch.cos(phi)), dim=-1)  # [B,2]

        rff = self.rff(t2d)  # [B, rff_out_dim]
        return self.mlp(rff)  # [B, emb_dim]


class GeoCLIPTimeEncoder(nn.Module):
    """
    Wraps rshf.GeoCLIP *without* touching its internals.
    Adds a small temporal RFF+MLP branch and fuses at the embedding level.

    Fusion is controlled via `fusion_mode`:
      - 'add'
      - 'hadamard'
      - 'concat' (uses a fusion MLP to go back to emb_dim)
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        emb_dim: int,
        time_variant: str = "doy",
        fusion_mode: str = "had",
        temporal_sigma: Union[float, Sequence[float]] = (2.0, 4.0),
        temporal_encoded_size: int = 64,
        temporal_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.emb_dim = emb_dim

        self.time_variant = time_variant.lower()
        self.use_time = ("doy" in self.time_variant) or ("month" in self.time_variant)

        self.fusion_mode = fusion_mode.lower()
        if self.fusion_mode not in ("add", "hadamard", "concat"):
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if self.use_time:
            self.time_encoder = TemporalRFFEncoder(
                emb_dim=emb_dim,
                sigma=temporal_sigma,
                encoded_size=temporal_encoded_size,
                hidden_dim=temporal_hidden_dim,
                time_variant=self.time_variant,
            )
        else:
            self.time_encoder = None

        if self.use_time and self.fusion_mode == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.fusion_mlp = None

    def forward(
        self,
        coords: torch.Tensor,
        time_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        coords  : [B, 2]  (lat, lon in degrees, as expected by GeoCLIP)
        time_idx: [B] or [B,1] (DOY or month, depending on time_variant)
        """
        # Spatial embedding from rshf GeoCLIP (unchanged)
        emb_spatial = self.base_encoder(coords)  # [B, emb_dim]

        if not self.use_time or time_idx is None:
            return emb_spatial

        emb_time = self.time_encoder(time_idx)  # [B, emb_dim]

        if self.fusion_mode == "hadamard":
            fused = emb_spatial * emb_time
        elif self.fusion_mode == "concat":
            fused = self.fusion_mlp(torch.cat([emb_spatial, emb_time], dim=-1))
        else:  # "add"
            fused = emb_spatial + emb_time

        return fused
