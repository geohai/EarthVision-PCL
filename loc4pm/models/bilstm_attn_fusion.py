"""BiLSTM with attention and location embedding fusion.

This module defines a PyTorch model that extends the baseline BiLSTM with
Luong-style attention by optionally incorporating a learned or pre-trained
geolocation encoder. The resulting location embedding can be fused with
the sequence context vector via concatenation or element-wise (Hadamard)
product prior to a small regression head.

The design is inspired by an earlier TensorFlow implementation that fuses
the outputs of a Bi-LSTM branch and a location encoder, with optional
projection and normalization of the location embedding. See the
configuration parameters for usage details.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .bilstm_attn import LuongAttention

_LOG = logging.getLogger(__name__)

class LocationEncoderWrapper(nn.Module):
    """Wrapper around optional external location encoders.

    This helper attempts to load a pre-trained geolocation encoder from the
    ``rshf`` package when requested. If the import fails or the encoder
    cannot be constructed, a simple MLP fallback is used to map the
    coordinate inputs (and optionally a month index) to a fixed-size
    embedding.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 variant: Optional[str] = None,
                 emb_dim: int = 256,
                 pretrained: bool = True,
                 freeze: bool = True,
                 project_dim: Optional[int] = None):
        super().__init__()
        self.name = name
        self.variant = variant
        self.emb_dim = emb_dim
        self.project_dim = project_dim
        self.pretrained = pretrained
        self.freeze = freeze

        self.encoder = None  # will hold external encoder if available
        encoder_loaded = False
        # Attempt to lazily import the requested encoder
        if name is not None and name.lower() not in ('none', ''):
            lname = name.lower()
            try:
                if lname == 'climplicit':
                    # Climplicit encoder has annual (1024-d) and monthly (256-d) modes.
                    from rshf.climplicit import Climplicit  # type: ignore
                    # The checkpoint names used here mirror those from the user's example.
                    # Users can override with different checkpoints via config if needed.
                    model_name = "Jobedo/climplicit"
                    cfg = {"return_chelsa": False}
                    if pretrained:
                        self.encoder = Climplicit.from_pretrained(model_name, config=cfg)
                    else:
                        self.encoder = Climplicit(config=cfg)
                    encoder_loaded = True
                elif lname == 'geoclip':
                    # GeoCLIP expects coordinates in [lat, lon] order.
                    try:
                        from rshf.geoclip import GeoCLIP  # type: ignore
                    except Exception:
                        from rshf.climplicit import GeoCLIP  # type: ignore
                    model_name = "MVRL/geoclip-location-encoder"
                    if pretrained:
                        self.encoder = GeoCLIP.from_pretrained(model_name)
                    else:
                        self.encoder = GeoCLIP()
                    encoder_loaded = True
                elif lname == 'satclip':
                    # SatCLIP expects coordinates in [lon, lat] order.
                    from rshf.satclip import SatClip  # type: ignore
                    model_name = "MVRL/satclip-loc-enc-vit16-l40"
                    if pretrained:
                        self.encoder = SatClip.from_pretrained(model_name)
                    else:
                        self.encoder = SatClip()
                    encoder_loaded = True
            except ImportError:
                _LOG.warning(
                    "Could not import rshf package; falling back to a simple MLP for location encoding."
                )
            except Exception as e:
                _LOG.warning(
                    "Failed to initialize %s encoder (%s); falling back to a simple MLP: %s",
                    name, variant, e
                )
        # If no external encoder was loaded, create a simple MLP fallback
        if not encoder_loaded or self.encoder is None:
            # Determine input dimensionality: lat+lon, plus optional month
            input_dim = 2 + (1 if variant and variant.lower() == 'monthly' else 0)
            hidden = max(emb_dim, 64)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, emb_dim)
            )
            encoder_loaded = False
        # Optionally freeze the underlying encoder parameters
        if self.encoder is not None and freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        # Optional projection layer: if provided, this maps the raw encoder output
        # to a user-specified dimension (useful when performing Hadamard fusion).
        if project_dim is not None and project_dim > 0 and project_dim != emb_dim:
            self.proj = nn.Linear(emb_dim, project_dim, bias=False)
            self.output_dim = project_dim
        else:
            self.proj = None
            self.output_dim = emb_dim
        # LayerNorm for the location embedding; helps stabilize training
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, coords: torch.Tensor, month: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute a location embedding.

        Args:
            coords: Tensor of shape [B, 2] containing latitude and longitude in degrees.
            month: Optional tensor of shape [B] or [B,1] containing month indices (1-12).

        Returns:
            Tensor of shape [B, output_dim] containing the location embedding.
        """
        if coords is None:
            raise ValueError("coords tensor must be provided for location encoding")
        # Ensure month is a column vector if provided
        if month is not None:
            if month.dim() != 1:
                month = month.squeeze()
            month = month.to(dtype=coords.dtype)
        lname = (self.name or '').lower()
        # Use external encoder if available
        if lname == 'climplicit':
            # Climplicit expects [lon, lat] order
            xy = torch.stack([coords[:,1], coords[:,0]], dim=1)
            if self.variant and self.variant.lower() == 'monthly' and month is not None:
                out = self.encoder(xy, month)
            else:
                out = self.encoder(xy)
        elif lname == 'geoclip':
            # GeoCLIP expects [lat, lon]
            out = self.encoder(coords)
        elif lname == 'satclip':
            # SatCLIP expects [lon, lat]
            xy = torch.stack([coords[:,1], coords[:,0]], dim=1)
            out = self.encoder(xy)
        else:
            # Fallback MLP: concatenate coords and (optionally) month
            if month is not None:
                inp = torch.cat([coords, month], dim=1)
            else:
                inp = coords
            out = self.encoder(inp)
        # Project if required
        if self.proj is not None:
            out = self.proj(out)
        # Normalize
        out = self.norm(out)
        return out


class BiLSTMAttnLocRegressor(nn.Module):
    """Bi-LSTM regressor with attention and optional location fusion.

    This model extends the base BiLSTM regressor by fusing a location
    embedding with the LSTM's attention-derived context vector. Fusion is
    either via element-wise multiplication (Hadamard) or concatenation.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 layer_norm: bool = True,
                 attn_type: str = 'luong',
                 attn_dim: int = 256,
                 *,
                 loc_name: Optional[str] = None,
                 loc_variant: Optional[str] = None,
                 loc_emb_dim: int = 256,
                 loc_pretrained: bool = True,
                 loc_freeze: bool = True,
                 loc_proj_dim: Optional[int] = None,
                 fusion_method: str = 'concat',
                 fusion_hidden_dim: int = 256):
        super().__init__()
        # Sequence branch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        # Attention mechanism
        self.attn = LuongAttention(out_dim, attn_dim if attn_type=='luong' else None)
        # Location encoder (optional)
        self.loc_encoder = None
        if loc_name and str(loc_name).lower() not in ('none', ''):
            self.loc_encoder = LocationEncoderWrapper(
                name=loc_name,
                variant=loc_variant,
                emb_dim=loc_emb_dim,
                pretrained=loc_pretrained,
                freeze=loc_freeze,
                project_dim=loc_proj_dim
            )
            loc_out_dim = self.loc_encoder.output_dim
        else:
            loc_out_dim = 0
        # 1) Temporal context dimension (BiLSTM output per time step)
        self.ts_ctx_dim = hidden_size * (2 if bidirectional else 1)

        # 2) Location embedding dimension after optional projection
        loc_dim = loc_proj_dim if loc_proj_dim is not None else loc_emb_dim
        # 3) Fused dimension
        fm = fusion_method.lower() if fusion_method else 'concat'
        if fm == 'concat':
            fused_in = self.ts_ctx_dim + loc_dim
        elif fm == 'hadamard':
            if loc_dim != self.ts_ctx_dim:
                raise ValueError(
                    f"Hadamard fusion requires equal dims: ts={self.ts_ctx_dim}, loc={loc_dim}. "
                    "Set model.location.proj_dim to match ts_ctx_dim."
                )
            fused_in = self.ts_ctx_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")
        self.fusion_method = fm
        # Regression head: map fused representation to a scalar via a hidden layer
        self.head = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1)
        )

    def forward(self,
                x: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                month: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Tensor of shape [B,T,F] containing the time-series input.
            coords: Optional tensor of shape [B,2] with (lat, lon) per sample.
            month: Optional tensor of shape [B] or [B,1] with month indices (1-12).
            mask: Optional tensor indicating valid positions in ``x`` (unused in current implementation).

        Returns:
            tuple: (y_hat, attn) where ``y_hat`` is the predicted scalar for each
            sample and ``attn`` are the attention weights over the time dimension.
        """
        # Sequence branch
        out, (h, _) = self.lstm(x)
        out = self.norm(out)
        # Build query from the last hidden states
        if self.bidirectional:
            q = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            q = h[-1]
        ctx, attn = self.attn(out, q, mask)
        # Optionally compute location embedding
        loc_emb = None
        if self.loc_encoder is not None:
            if coords is None:
                raise ValueError("coords must be provided when using a location encoder")
            loc_emb = self.loc_encoder(coords, month)
        # Fuse context and location
        if self.fusion_method == 'hadamard' and loc_emb is not None:
            fused = ctx * loc_emb
        elif loc_emb is not None:
            fused = torch.cat([ctx, loc_emb], dim=-1)
        else:
            fused = ctx
        y = self.head(fused).squeeze(-1)
        return y, attn
