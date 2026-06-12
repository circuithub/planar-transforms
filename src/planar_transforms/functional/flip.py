"""Field-typed flip: reflect a field across a spatial axis.

Dispatched on the field type like :func:`planar_transforms.functional.rotate`.
Continuous and discrete fields reverse the spatial index, which is exact for both (no
interpolation, so discrete labels are never blended). Sets reflect the flipped r2 axis;
oriented boxes also conjugate the rotor so the box orientation mirrors with the points.

``axis="horizontal"`` mirrors left-right (reverses image columns / r2 x); ``axis="vertical"``
mirrors top-bottom (reverses image rows / r2 y). An optional ``mask`` of shape ``(B,)``
selects which batch elements flip, leaving the rest untouched (random per-sample flip).
"""

from typing import Literal, Optional, TypeVar

import torch

from planar_transforms import r2
from planar_transforms.types import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    PointSet,
    VectorSet,
)

T = TypeVar("T", ContinuousField, DiscreteField, VectorSet, PointSet, BoxSet)

Axis = Literal["horizontal", "vertical"]


def _as_mask(mask: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    return mask.expand(batch_size)


def flip(
    x: T,
    axis: Axis = "horizontal",
    mask: Optional[torch.Tensor] = None,
) -> T:
    """Reflect a batched field across ``axis``, optionally per-sample via ``mask``.

    ``mask`` is a length-B boolean tensor (``True`` flips that sample); ``None`` flips the
    whole batch. The field type sets the behaviour: continuous/discrete fields reverse the
    spatial index, sets reflect the flipped r2 coordinate, oriented boxes also conjugate
    the rotor.
    """
    if axis not in ("horizontal", "vertical"):
        raise ValueError(f"axis must be 'horizontal' or 'vertical', got {axis!r}")

    if isinstance(x, (ContinuousField, DiscreteField)):
        spatial_dim = -1 if axis == "horizontal" else -2
        flipped = x.flip(dims=(spatial_dim,))
        sample_mask = _as_mask(mask, x.size(dim=0), x.device)
        # Broadcast the (B,) mask over C, H, W.
        sample_mask = sample_mask.view(-1, *([1] * (x.dim() - 1)))
        return type(x)(torch.where(sample_mask, flipped, x))

    if isinstance(x, (r2.VectorSet, r2.PointSet, r2.BoxSet, r2.OrientedBoxSet)):
        # r2 x <-> image columns (horizontal), r2 y <-> image rows (vertical).
        axis_index = 0 if axis == "horizontal" else 1
        sample_mask = _as_mask(mask, x.size(dim=0), x.device)
        sign = torch.ones(x.size(dim=0), 2, device=x.device, dtype=x.dtype)
        sign[sample_mask, axis_index] = -1.0
        for _ in range(x.dim() - 2):
            sign = sign[:, None, ...]

        if isinstance(x, (r2.VectorSet, r2.PointSet)):
            return type(x)(x * sign)
        if isinstance(x, r2.OrientedBoxSet):
            # A reflection conjugates the rotor (negate the sin/bivector component),
            # mirroring orientation for either axis. Render-verified in
            # tests/test_image_geometry_alignment.py.
            rotor_sign = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
            flipped_rotors = x.rotors * torch.where(
                sample_mask.view(-1, 1),
                rotor_sign,
                torch.ones_like(rotor_sign),
            )
            return r2.OrientedBoxSet(
                radii=x.radii,
                centroids=x.centroids * sign,
                rotors=flipped_rotors,
            )
        if isinstance(x, r2.BoxSet):
            return r2.BoxSet(radii=x.radii, centroids=x.centroids * sign)
        assert False, "Unhandled type"

    raise NotImplementedError(f"Unsupported input type {type(x)} for the flip transform.")
