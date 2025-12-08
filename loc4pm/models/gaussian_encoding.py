import math
from typing import Iterable, List, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["GaussianEncoding"]


class GaussianEncoding(nn.Module):
    """
    Approximate Gaussian random Fourier feature encoding for continuous inputs.

    This module implements a simple Gaussian random Fourier feature encoding
    inspired by GeoCLIP and related location encoders.  Given an input tensor
    ``x`` of shape ``[B, input_size]`` it projects ``x`` using one or more
    randomly sampled Gaussian matrices and returns the concatenation of the
    corresponding sinusoidal basis functions.

    Parameters
    ----------
    sigma : float or Sequence[float]
        Standard deviation(s) of the Gaussian kernel(s).  When a sequence is
        provided the encoder draws a separate set of random projections for
        each ``sigma`` and concatenates the resulting encodings.  Larger
        values of ``sigma`` correspond to lower frequency (smoother) features.
    input_size : int
        Dimensionality of the input coordinates (e.g. 2 for latitude/longitude
        or 1 for day-of-year).
    encoded_size : int
        Number of random features to draw *per sigma*.  The final output
        dimension before the sinusoidal doubling is ``encoded_size``; after
        applying ``sin`` and ``cos`` and concatenating across ``sigma`` the
        output dimension becomes ``2 * encoded_size * len(sigma)``.

    Notes
    -----
    The random weight matrices are sampled from a zero mean Gaussian with
    variance ``1 / sigma**2`` along each dimension.  At inference time the
    projection matrices are fixed.  A scaling factor of ``math.sqrt(2.0)
    / encoded_size`` is applied so that the resulting features have roughly
    unit variance.
    """

    def __init__(
        self,
        sigma: Union[float, Sequence[float]] = 1.0,
        input_size: int = 2,
        encoded_size: int = 256,
    ) -> None:
        super().__init__()
        # Normalize sigma to a list for uniform handling
        if isinstance(sigma, (list, tuple)):
            self.sigmas: List[float] = [float(s) for s in sigma]
        else:
            self.sigmas = [float(sigma)]
        self.input_size = int(input_size)
        self.encoded_size = int(encoded_size)
        # For each sigma draw a weight matrix from N(0, 1/sigma^2)
        # Shape: [len(sigmas), input_size, encoded_size]
        weight_list = []
        for s in self.sigmas:
            # variance of projection weights is 1/s^2 so std is 1/s
            w = torch.randn(self.input_size, self.encoded_size) / float(s)
            weight_list.append(w)
        # Register as buffer so they are not trainable but move with the module
        self.register_buffer("weight", torch.stack(weight_list, dim=0))
        # Scaling factor for the final encoding; using sqrt(2) so that sin and cos
        # components each have unit variance when summed
        self.scale = math.sqrt(2.0 / self.encoded_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian random Fourier features for input ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, input_size]``.

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape ``[B, 2 * encoded_size * len(sigmas)]``.
        """
        if x.dim() != 2 or x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input of shape [B, {self.input_size}], got {tuple(x.shape)}"
            )
        # x shape: [B, input_size]
        # weight shape: [S, input_size, encoded_size]
        # project x for each sigma: x_proj shape [S, B, encoded_size]
        # We compute via einsum for efficiency
        # x_proj[s] = x @ weight[s]
        x_proj = torch.einsum("bi,sio->sbo", x, self.weight)
        # Apply sinusoidal functions and concatenate sin and cos features
        # sin/cos expect [S, B, encoded_size]; output [S, B, encoded_size]
        sin = torch.sin(x_proj)
        cos = torch.cos(x_proj)
        # Flatten along sigma dimension and encoded dimension: [B, S * encoded_size]
        sin_flat = sin.permute(1, 0, 2).reshape(x.size(0), -1)
        cos_flat = cos.permute(1, 0, 2).reshape(x.size(0), -1)
        # Scale features
        features = self.scale * torch.cat([cos_flat, sin_flat], dim=-1)
        return features.float()