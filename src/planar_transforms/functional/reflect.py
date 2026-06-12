"""Field-typed reflect: reflect a field across a spatial axis.

Dispatched on the field type like :func:`planar_transforms.functional.rotate`.
Continuous and discrete fields reverse the spatial index, which is exact for both (no
interpolation, so discrete labels are never blended). Sets reflect the named r2
coordinate; oriented boxes also conjugate the rotor so the box orientation mirrors with
the points.

``axis`` names the *coordinate reflected*, like ``torch.flip`` dims -- not the mirror
line. ``axis="x"`` negates r2 x / reverses image columns (left-right mirror);
``axis="y"`` negates r2 y / reverses image rows (top-bottom mirror). An optional ``mask``
of shape ``(B,)`` selects which batch elements reflect, leaving the rest untouched
(random per-sample reflect).

``reflect`` is intentionally a dedicated exact op (index reversal), not routed through
``affine``/``grid_sample``, for exactness and lower cost.
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

Axis = Literal["x", "y"]


def _as_mask(mask: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    return mask.expand(batch_size)


def reflect(
    x: T,
    axis: Axis = "x",
    mask: Optional[torch.Tensor] = None,
) -> T:
    """Reflect a batched field across ``axis``, optionally per-sample via ``mask``.

    ``axis`` names the coordinate reflected (like ``torch.flip`` dims, not the mirror
    line): ``"x"`` negates r2 x / reverses image columns (left-right mirror), ``"y"``
    negates r2 y / reverses image rows (top-bottom mirror). ``mask`` is a length-B boolean
    tensor (``True`` reflects that sample); ``None`` reflects the whole batch. The field
    type sets the behaviour: continuous/discrete fields reverse the spatial index, sets
    reflect the named r2 coordinate, oriented boxes also conjugate the rotor.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    if isinstance(x, (ContinuousField, DiscreteField)):
        spatial_dim = -1 if axis == "x" else -2
        reflected = x.flip(dims=(spatial_dim,))
        sample_mask = _as_mask(mask, x.size(dim=0), x.device)
        # Broadcast the (B,) mask over C, H, W.
        sample_mask = sample_mask.view(-1, *([1] * (x.dim() - 1)))
        return type(x)(torch.where(sample_mask, reflected, x))

    if isinstance(x, (r2.VectorSet, r2.PointSet, r2.BoxSet, r2.OrientedBoxSet)):
        # r2 x <-> image columns, r2 y <-> image rows.
        axis_index = 0 if axis == "x" else 1
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
            reflected_rotors = x.rotors * torch.where(
                sample_mask.view(-1, 1),
                rotor_sign,
                torch.ones_like(rotor_sign),
            )
            return r2.OrientedBoxSet(
                radii=x.radii,
                centroids=x.centroids * sign,
                rotors=reflected_rotors,
            )
        if isinstance(x, r2.BoxSet):
            return r2.BoxSet(radii=x.radii, centroids=x.centroids * sign)
        assert False, "Unhandled type"

    raise NotImplementedError(f"Unsupported input type {type(x)} for the reflect transform.")
