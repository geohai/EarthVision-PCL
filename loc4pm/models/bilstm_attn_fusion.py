"""BiLSTM with attention, location fusion and dual physical heads for LOC4PM.

This module extends the original ``BiLSTMAttnLocRegressor`` by adding support
for an auxiliary physical head.  The primary physical head predicts the
main physical target (e.g. PM2.5), while the auxiliary head predicts
additional physical variables such as meteorological conditions (e.g.
``AIR_DENS``, ``RH``, ``SFC_TMP``, etc.).  Both heads operate on the
location embedding produced by the location encoder.  The observation
head remains unchanged and predicts the pollution observation ``y`` from
the fused time-series and location embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .bilstm_attn import LuongAttention
from .siren import DirectSirenEncoder
from .rff_encoder import GeoCLIPTimeEncoder

_LOG = logging.getLogger(__name__)

__all__ = ["BiLSTMAttnLocRegressor", "LocationEncoderWrapper"]


class LocationEncoderWrapper(nn.Module):
    """Wrapper around optional external or local location encoders.

    This helper attempts to construct a location encoder according to
    user configuration.  Supported encoders include: ``climplicit``,
    ``geoclip``, ``satclip``, ``siren``, ``resiren``, and ``geoclip-time``.
    If the specified encoder cannot be loaded, a fallback MLP is used.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        variant: Optional[str] = None,
        emb_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        project_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.variant = variant
        self.emb_dim = emb_dim
        self.project_dim = project_dim
        self.pretrained = pretrained
        self.freeze = freeze
        self.encoder: Optional[nn.Module] = None
        encoder_loaded = False
        # Attempt to load requested encoder
        if name is not None and str(name).lower() not in ('none', ''):
            lname = str(name).lower()
            try:
                if lname == 'climplicit':
                    from rshf.climplicit import Climplicit  # type: ignore
                    model_name = "Jobedo/climplicit"
                    cfg = {"return_chelsa": False}
                    if pretrained:
                        self.encoder = Climplicit.from_pretrained(model_name, config=cfg)
                    else:
                        self.encoder = Climplicit(config=cfg)
                    encoder_loaded = True
                elif lname == 'geoclip':
                    from rshf.geoclip import GeoCLIP, GeoCLIPConfig  # type: ignore
                    model_name = "MVRL/geoclip-location-encoder"
                    if pretrained:
                        self.encoder = GeoCLIP.from_pretrained(model_name)
                    else:
                        self.encoder = GeoCLIP(GeoCLIPConfig(sigma=[2, 2**2], input_size=2, encoded_size=256, dim=512))
                    encoder_loaded = True
                elif lname == 'satclip':
                    from rshf.satclip import SatClip  # type: ignore
                    model_name = "MVRL/satclip-loc-enc-vit16-l40"
                    if pretrained:
                        self.encoder = SatClip.from_pretrained(model_name)
                    else:
                        self.encoder = SatClip()
                    encoder_loaded = True
                elif lname in ('siren', 'resiren'):
                    var = str(self.variant).lower() if self.variant is not None else None
                    monthly_flag = (var == 'monthly') if var is not None else bool(self.variant and str(self.variant).lower() == 'monthly')
                    self.encoder = DirectSirenEncoder(
                        emb_dim=emb_dim,
                        dim_hidden=max(emb_dim, 512),
                        num_layers=16,
                        residual=(lname == 'resiren'),
                        h_siren=True,
                        w0=1.0,
                        w0_initial=30.0,
                        monthly=monthly_flag,
                        variant=var,
                    )
                    encoder_loaded = True
                elif lname in ('geoclip-time', 'geoclip_time'):
                    from rshf.geoclip import GeoCLIP, GeoCLIPConfig  # type: ignore
                    model_name = "MVRL/geoclip-location-encoder"
                    if pretrained:
                        base = GeoCLIP.from_pretrained(model_name)
                    else:
                        base = GeoCLIP(GeoCLIPConfig(sigma=[2, 2 ** 2], input_size=2, encoded_size=256, dim=512))
                    if freeze:
                        for p in base.parameters():
                            p.requires_grad = False
                    v = (variant or "").lower()
                    if "had" in v:
                        fusion_mode = "hadamard"
                    elif "cat" in v:
                        fusion_mode = "concat"
                    else:
                        fusion_mode = "add"
                    self.encoder = GeoCLIPTimeEncoder(
                        base_encoder=base,
                        emb_dim=emb_dim,
                        time_variant=v or "doy",
                        fusion_mode=fusion_mode,
                        temporal_sigma=[1.0, 2.0],
                        temporal_encoded_size=64,
                        temporal_hidden_dim=256,
                    )
                    encoder_loaded = True
            except ImportError:
                _LOG.warning(
                    "Could not import requested location encoder '%s'; falling back to a simple MLP.",
                    lname,
                )
            except Exception as e:
                _LOG.warning(
                    "Failed to initialize %s encoder (%s); falling back to a simple MLP: %s",
                    name,
                    variant,
                    e,
                )
        # Fallback MLP if no external encoder was loaded
        if not encoder_loaded or self.encoder is None:
            var_l = str(self.variant).lower() if self.variant is not None else None
            has_temporal = var_l in ('monthly', 'doy') if var_l is not None else False
            input_dim = 2 + (1 if has_temporal else 0)
            hidden_dim = max(emb_dim, 64)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, emb_dim),
            )
            encoder_loaded = False
        # Freeze parameters if requested
        if self.encoder is not None and freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        # Optional projection
        if project_dim is not None and project_dim > 0:
            self.proj = nn.Linear(emb_dim, project_dim, bias=False)
            self.output_dim = project_dim
        else:
            self.proj = None
            self.output_dim = emb_dim
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, coords: torch.Tensor, month: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords tensor must be provided for location encoding")
        # Ensure month is a column vector if provided
        if month is not None:
            if month.dim() != 1:
                month = month.squeeze()
            month = month.to(dtype=coords.dtype)
        lname = (self.name or '').lower()
        if lname == 'climplicit':
            xy = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
            if self.variant and str(self.variant).lower() == 'monthly' and month is not None:
                out = self.encoder(xy, month)
            else:
                out = self.encoder(xy)
        elif lname == 'geoclip':
            out = self.encoder(coords)
        elif lname == 'satclip':
            xy = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
            xy = xy.double()
            out = self.encoder(xy)
            out = out.float()
        elif lname in ('siren', 'resiren'):
            out = self.encoder(coords, month)
        elif lname in ('geoclip-time', 'geoclip_time'):
            out = self.encoder(coords, month)
        else:
            if month is not None:
                inp = torch.cat([coords, month.unsqueeze(-1)], dim=1)
            else:
                inp = coords
            out = self.encoder(inp)
        if self.proj is not None:
            out = self.proj(out)
        out = self.norm(out)
        return out


class BiLSTMAttnLocRegressor(nn.Module):
    """Bi‑LSTM regressor with attention, location fusion, and dual physical heads.

    In addition to predicting the observation target ``y`` from the fused
    time-series and location embedding, this model optionally predicts a
    primary physical vector ``z`` via a ``physical_head`` and an auxiliary
    physical vector ``z_aux`` via ``aux_physical_head``.  Both heads map
    from the location embedding to their respective output dimensions.
    """

    def __init__(
        self,
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
        fusion_hidden_dim: int = 256,
        physical_head_hidden_dim: Optional[int] = None,
        physical_out_dim: int = 0,
        # new parameters for auxiliary physical head
        aux_physical_head_hidden_dim: Optional[int] = None,
        aux_physical_out_dim: int = 0,
        head_dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        # Sequence branch
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.ts_ctx_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(self.ts_ctx_dim) if layer_norm else nn.Identity()
        self.attn = LuongAttention(self.ts_ctx_dim, attn_dim if attn_type == 'luong' else None)
        self.loc_encoder: Optional[LocationEncoderWrapper] = None
        if loc_name and str(loc_name).lower() not in ('none', ''):
            self.loc_encoder = LocationEncoderWrapper(
                name=loc_name,
                variant=loc_variant,
                emb_dim=loc_emb_dim,
                pretrained=loc_pretrained,
                freeze=loc_freeze,
                project_dim=loc_proj_dim,
            )
            loc_emb_out = self.loc_encoder.output_dim
        else:
            loc_emb_out = 0
        # Determine location embedding dimensionality after projection
        loc_dim = loc_proj_dim if (loc_proj_dim is not None) else loc_emb_dim
        fm = fusion_method.lower() if fusion_method else 'concat'
        if fm == 'concat':
            fused_in = self.ts_ctx_dim + loc_dim
        elif fm == 'hadamard':
            if loc_dim != self.ts_ctx_dim:
                raise ValueError(
                    f"Hadamard fusion requires equal dims: ts_ctx_dim={self.ts_ctx_dim}, loc_dim={loc_dim}. "
                    "Set model.location.proj_dim equal to the LSTM context dimension."
                )
            fused_in = self.ts_ctx_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        self.fusion_method = fm
        hd = float(dropout) if head_dropout is None else float(head_dropout)
        self.head = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(hd),
            nn.Linear(fusion_hidden_dim, 1),
        )
        # Primary physical head
        self.physical_head: Optional[nn.Module] = None
        if physical_out_dim and physical_head_hidden_dim is not None and physical_out_dim > 0:
            self.physical_head = nn.Sequential(
                nn.Linear(loc_dim, physical_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(physical_head_hidden_dim, physical_out_dim),
            )
        self.physical_out_dim = physical_out_dim
        # Auxiliary physical head
        self.aux_physical_head: Optional[nn.Module] = None
        if aux_physical_out_dim and aux_physical_head_hidden_dim is not None and aux_physical_out_dim > 0:
            self.aux_physical_head = nn.Sequential(
                nn.Linear(loc_dim, aux_physical_head_hidden_dim),
                nn.ReLU(),
                nn.Linear(aux_physical_head_hidden_dim, aux_physical_out_dim),
            )
        self.aux_physical_out_dim = aux_physical_out_dim

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        # Sequence branch
        out, (h, _) = self.lstm(x)
        out = self.norm(out)
        if self.bidirectional:
            q = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            q = h[-1]
        ctx, attn = self.attn(out, q, mask)
        loc_emb: Optional[torch.Tensor] = None
        if self.loc_encoder is not None:
            if coords is None:
                raise ValueError("coords must be provided when using a location encoder")
            loc_emb = self.loc_encoder(coords, month)
        if self.fusion_method == 'hadamard' and loc_emb is not None:
            fused = ctx * loc_emb
        elif loc_emb is not None:
            fused = torch.cat([ctx, loc_emb], dim=-1)
        else:
            fused = ctx
        y_hat = self.head(fused).squeeze(-1)
        z_hat: Optional[torch.Tensor] = None
        if self.physical_head is not None and loc_emb is not None:
            z_hat = self.physical_head(loc_emb)
        aux_z_hat: Optional[torch.Tensor] = None
        if self.aux_physical_head is not None and loc_emb is not None:
            aux_z_hat = self.aux_physical_head(loc_emb)
        return y_hat, z_hat, aux_z_hat, attn