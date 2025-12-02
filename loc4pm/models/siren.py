"""SIREN location encoder modules for LOC4PM.

This module implements sinusoidal representation networks (SIREN) and their
residual variant (ReSIREN) as well as a convenience wrapper for creating a
location encoder that optionally augments normalized coordinates with a
temporal sinusoidal input.  The temporal input can be monthly (1–12) or
day‑of‑year (1–365) and is encoded via ``sin`` and ``cos`` of the appropriate
phase.  See the original SIREN paper (Sitzmann et al., 2020) for details on
the architecture.
"""

from __future__ import annotations

from typing import Iterable, Optional

import math
import torch
from torch import nn
import torch.nn.functional as F

from .direct import Direct

__all__ = ["SirenNet", "Siren", "Sine", "DirectSirenEncoder"]


class Sine(nn.Module):
    """Sine activation used in SIREN layers.

    Parameters
    ----------
    w0 : float, default=1.0
        Frequency scaling factor applied to the input before the sine.
    """

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """Single SIREN layer.

    A SIREN layer applies a linear transformation followed by a sinusoidal
    activation.  It supports optional residual connections (as used in
    ReSIREN) by averaging the pre‑activation output with a Gaussian from
    the previous layer.  When ``h_siren`` is True, the first layer uses a
    hyperbolic sine (sinh) activation instead of a plain sine, as
    described in the H‑SIREN variant.

    Parameters
    ----------
    dim_in : int
        Input dimensionality.
    dim_out : int
        Output dimensionality.
    w0 : float, default=1.0
        Base frequency of the sine activation for layers beyond the first.
    c : float, default=6.0
        Scaling constant for the weight initialization.
    is_first : bool, default=False
        Whether this layer is the first layer in the network.  First
        layers receive a higher frequency (``w0_initial``) and different
        initialization.
    use_bias : bool, default=True
        Whether to include a learnable bias term.
    activation : Optional[nn.Module], default=None
        Optional custom activation.  If ``None``, a sine with frequency
        ``w0`` is used.
    dropout : bool, default=False
        Whether to apply dropout after the linear transform.
    residual_connections : bool, default=False
        If True, the layer expects a ``prev_gaussian`` tensor when
        called and averages it with the current Gaussian before
        activation, forming a residual connection across layers.
    h_siren : bool, default=False
        Whether to use the hyperbolic sine activation on the first layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: Optional[nn.Module] = None,
        dropout: bool = False,
        residual_connections: bool = False,
        h_siren: bool = False,
    ) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.is_first = bool(is_first)
        self.dropout = bool(dropout)
        self.residual_connections = bool(residual_connections)
        self.h_siren = bool(h_siren)

        # Initialize weight and bias with appropriate scale
        weight = torch.empty(self.dim_out, self.dim_in)
        bias = torch.empty(self.dim_out) if use_bias else None
        self._init_weights(weight, bias, c=c, w0=w0)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

        # Choose activation: default is sine
        self.activation: nn.Module = activation if activation is not None else Sine(w0)

    def _init_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor], *, c: float, w0: float) -> None:
        """Initialize weights following the SIREN initialization scheme."""
        dim = self.dim_in
        # Use a higher standard deviation for the first layer
        w_std = (1.0 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x: torch.Tensor, prev_gaussian: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Linear transform
        out = F.linear(x, self.weight, self.bias)
        # Optional dropout for regularization
        if self.dropout:
            out = F.dropout(out, training=self.training)
        # Merge with previous Gaussian for residual connections
        gaussian = out
        if self.residual_connections and prev_gaussian is not None and gaussian.shape == prev_gaussian.shape:
            gaussian = (gaussian + prev_gaussian) / 2.0
        # Apply H‑SIREN (sinh) on first layer if enabled
        if self.h_siren and self.is_first:
            out = torch.sinh(2.0 * gaussian)
        else:
            out = gaussian
        # Apply activation
        out = self.activation(out)
        return out, gaussian


class SirenNet(nn.Module):
    """Stack of SIREN layers culminating in a linear projection.

    This module constructs a network with ``num_layers`` sinusoidal layers
    (instances of :class:`Siren`) followed by a final linear layer.  It
    optionally returns intermediate hidden embeddings when
    ``return_hidden_embs`` is a sequence of layer indices.  Setting
    ``residual_connections`` enables the ReSIREN variant.

    Parameters
    ----------
    dim_in : int
        Dimensionality of the input.
    dim_hidden : int
        Hidden dimension for the intermediate SIREN layers.
    dim_out : int
        Dimension of the output embedding.
    num_layers : int
        Number of hidden SIREN layers prior to the final linear layer.
    w0 : float, default=1.0
        Frequency of the sine activation for layers beyond the first.
    w0_initial : float, default=30.0
        Frequency of the sine activation for the first layer.  Higher values
        allow the network to represent higher‑frequency signals.
    use_bias : bool, default=True
        Whether to include biases in the linear transformations.
    final_activation : Optional[nn.Module], default=None
        Activation to apply to the output of the final layer.  If ``None``,
        no activation is applied (identity).
    dropout : bool, default=False
        If True, apply dropout in the hidden SIREN layers.
    residual_connections : bool, default=False
        Whether to average Gaussians across layers (ReSIREN).
    h_siren : bool, default=False
        Whether to use H‑SIREN activation for the first layer.
    return_hidden_embs : Optional[Iterable[int]], default=None
        Indices of hidden layers whose embeddings should be returned.  If
        provided, the final output is a concatenation of these hidden
        embeddings and the final layer output.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: Optional[nn.Module] = None,
        dropout: bool = False,
        residual_connections: bool = False,
        h_siren: bool = False,
        return_hidden_embs: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "SirenNet must have at least one hidden layer"
        self.num_layers = int(num_layers)
        self.dim_hidden = int(dim_hidden)
        self.return_hidden_embs = list(return_hidden_embs) if return_hidden_embs is not None else None
        self.residual_connections = bool(residual_connections)
        self.h_siren = bool(h_siren)

        # Build hidden SIREN layers
        self.layers = nn.ModuleList()
        for idx in range(self.num_layers):
            is_first = idx == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                    residual_connections=self.residual_connections,
                    h_siren=self.h_siren,
                )
            )

        # Final linear layer; apply optional activation afterwards
        self.final_activation: nn.Module = final_activation if final_activation is not None else nn.Identity()
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=self.final_activation,
            dropout=False,
        )

    def forward(self, x: torch.Tensor, mods: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the SIREN embedding for the input.

        Args
        ----
        x : torch.Tensor
            Input tensor of shape ``[B, dim_in]``.
        mods : Optional[torch.Tensor]
            Unused placeholder for compatibility with some HuggingFace model
            signatures.  Ignored in this implementation.

        Returns
        -------
        torch.Tensor
            The output embedding.  If ``return_hidden_embs`` was provided,
            returns the concatenation of the selected hidden embeddings and
            the final output.  Otherwise returns only the final output.
        """
        res_outputs: list[torch.Tensor] = []
        prev_gaussian: Optional[torch.Tensor] = None
        for idx, layer in enumerate(self.layers):
            x, gaussian = layer(x, prev_gaussian)
            prev_gaussian = gaussian
            if self.return_hidden_embs is not None and idx in self.return_hidden_embs:
                res_outputs.append(x)
        x, _ = self.last_layer(x, prev_gaussian)
        if res_outputs:
            res_outputs.append(x)
            x = torch.cat(res_outputs, dim=-1)
        return x


class DirectSirenEncoder(nn.Module):
    """
    rshf‑style location encoder with optional temporal sinusoidal input.

    This encoder normalizes geographic coordinates into the range ``[-1,1]``
    using :class:`Direct` and feeds them through a :class:`SirenNet`.  When
    configured with a temporal variant, it appends a two‑dimensional
    sinusoidal encoding of a temporal index to the spatial coordinates.  The
    supported temporal variants are ``"monthly"`` (month index 1–12) and
    ``"doy"`` (day‑of‑year index 1–365).  If neither variant is enabled,
    only the spatial coordinates are used.

    Parameters
    ----------
    emb_dim : int, default=256
        Output embedding dimension produced by the internal :class:`SirenNet`.
    dim_hidden : int, default=512
        Hidden dimension inside the :class:`SirenNet`.
    num_layers : int, default=16
        Number of SIREN layers.
    residual : bool, default=True
        If ``True``, enable residual (ReSIREN) connections between SIREN layers.
    h_siren : bool, default=True
        If ``True``, use hyperbolic sine (sinh) on the first layer as in H‑SIREN.
    w0 : float, default=1.0
        Base frequency for SIREN activations beyond the first layer.
    w0_initial : float, default=10.0
        Frequency for the first SIREN layer.
    monthly : bool, default=False
        Legacy flag to enable the ``monthly`` variant.  When ``variant`` is
        provided this flag is ignored.
    variant : Optional[str], default=None
        Optional temporal variant.  Accepts ``"monthly"`` or ``"doy"`` to
        specify the temporal encoding.  Overrides the ``monthly`` flag.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        dim_hidden: int = 512,
        num_layers: int = 16,
        residual: bool = True,
        h_siren: bool = True,
        w0: float = 1.0,
        w0_initial: float = 10.0,
        monthly: bool = False,
        variant: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Determine temporal variant.
        self.variant = str(variant).lower() if variant is not None else None
        if self.variant is None:
            self.monthly = False
            self.doy = False
        else:
            self.monthly = self.variant == "monthly"
            self.doy = self.variant == "doy"
        # Normalize lon/lat into [-1,1]
        self.pos = Direct(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)
        # Input dimension: 2 (lon,lat) plus 2 for sin/cos if a temporal variant is enabled
        dim_in = 4 if (self.monthly or self.doy) else 2
        # Construct SIREN network
        self.siren = SirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=emb_dim,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            residual_connections=residual,
            h_siren=h_siren,
        )

    def forward(self, coords: torch.Tensor, month: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute a SIREN location embedding for the given coordinates and optional temporal index.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape ``[B, 2]`` containing latitude and longitude in degrees
            as ``(lat, lon)``.  Internally, the tensor is reordered to ``(lon, lat)``
            for normalization.
        month : Optional[torch.Tensor], default=None
            Temporal index tensor.  When ``variant="monthly"`` this should contain
            month indices in 1–12.  When ``variant="doy"`` this should contain
            day‑of‑year indices in 1–365.  If no temporal variant is enabled this
            argument is ignored.

        Returns
        -------
        torch.Tensor
            Location embedding of shape ``[B, emb_dim]``.
        """
        if coords is None:
            raise ValueError("coords is required for DirectSirenEncoder")
        # Reorder to (lon, lat) for Direct scaling
        lonlat = torch.stack([coords[:, 1], coords[:, 0]], dim=1)
        loc = self.pos(lonlat)  # shape [B, 2]
        # Append temporal sin/cos features if enabled and provided
        if (self.monthly or self.doy) and (month is not None):
            # Flatten to [B]
            m = month.float().squeeze(-1) if month.dim() > 1 else month.float()
            # Choose denominator based on variant
            denom = 12.0 if self.monthly else 365.2425
            phi = m / denom * (2.0 * math.pi)
            # Concatenate sin and cos of phi
            loc = torch.cat([
                loc,
                torch.sin(phi).unsqueeze(-1),
                torch.cos(phi).unsqueeze(-1),
            ], dim=1)
        return self.siren(loc)