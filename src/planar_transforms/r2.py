"""Canonical r2 frame.

Continuous coordinates ordered ``(x, y)``, y increasing **up**, origin at the image
centre, in **pixel units** (isotropic). Geometric transforms operate here, with
textbook matrices (no transpose, no handedness fudging).

The component order is a deliberate flip from the pixel frame's ``(row, col)`` so a
forgotten conversion surfaces as obviously-wrong geometry rather than a silent swap.
The functions below are the only place axes are reinterpreted.

Frame map (size = ``(H, W)``), positions:
    x = col - (W-1)/2      y = (H-1)/2 - row
The origin is the image's rotation centre, ``((W-1)/2, (H-1)/2)`` (the centroid of
pixel indices), so rotation in r2 tracks the image rotation exactly. (Resize sampling
is a separate concern and uses the half-integer / align_corners=False convention.)
"""

import torch

from planar_transforms import pixel
from planar_transforms.types import BoxSet as _BoxSet
from planar_transforms.types import OrientedBoxSet as _OrientedBoxSet
from planar_transforms.types import PointSet as _PointSet
from planar_transforms.types import VectorSet as _VectorSet


class VectorSet(_VectorSet):
    """A displacement in r2 coordinates, ordered ``(dx, dy)``."""


class PointSet(_PointSet):
    """A position in r2 coordinates, ordered ``(x, y)``; origin image centre, y up."""


class BoxSet(_BoxSet):
    """Axis-aligned box: ``(radii_x, radii_y, centroid_x, centroid_y)``."""


class OrientedBoxSet(_OrientedBoxSet):
    """Oriented box; radii ``(x, y)``, rotor ``(cos, sin)`` half-angle, CCW."""


def _t(x):
    return x.as_subclass(torch.Tensor)


def _swap(ab):
    """Swap the last-dim pair (radii_row, radii_col) <-> (radii_x, radii_y)."""
    return torch.stack([ab[..., 1], ab[..., 0]], dim=-1)


def _point_pixel_to_r2(rowcol, H, W):
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    row, col = rowcol[..., 0], rowcol[..., 1]
    return torch.stack([col - cx, cy - row], dim=-1)


def _point_r2_to_pixel(xy, H, W):
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([cy - y, x + cx], dim=-1)


def _vec_pixel_to_r2(drowdcol):
    drow, dcol = drowdcol[..., 0], drowdcol[..., 1]
    return torch.stack([dcol, -drow], dim=-1)


def _vec_r2_to_pixel(dxdy):
    dx, dy = dxdy[..., 0], dxdy[..., 1]
    return torch.stack([-dy, dx], dim=-1)


def from_pixel(geom, size):
    """Convert pixel-frame geometry to the r2 frame. ``size`` is ``(H, W)``."""
    H, W = float(size[0]), float(size[1])
    if isinstance(geom, pixel.OrientedBoxSet):
        return OrientedBoxSet(
            radii=_swap(_t(geom.radii)),
            centroids=_point_pixel_to_r2(_t(geom.centroids), H, W),
            rotors=_t(geom.rotors).clone(),  # unchanged: net map is a rotation, not a reflection
        )
    if isinstance(geom, pixel.BoxSet):
        return BoxSet(
            radii=_swap(_t(geom.radii)),
            centroids=_point_pixel_to_r2(_t(geom.centroids), H, W),
        )
    if isinstance(geom, pixel.PointSet):
        return PointSet(_point_pixel_to_r2(_t(geom), H, W))
    if isinstance(geom, pixel.VectorSet):
        return VectorSet(_vec_pixel_to_r2(_t(geom)))
    raise NotImplementedError(f"from_pixel does not support {type(geom)}")


def to_pixel(geom, size):
    """Convert r2-frame geometry back to the pixel frame. ``size`` is ``(H, W)``."""
    H, W = float(size[0]), float(size[1])
    if isinstance(geom, OrientedBoxSet):
        return pixel.OrientedBoxSet(
            radii=_swap(_t(geom.radii)),
            centroids=_point_r2_to_pixel(_t(geom.centroids), H, W),
            rotors=_t(geom.rotors).clone(),
        )
    if isinstance(geom, BoxSet):
        return pixel.BoxSet(
            radii=_swap(_t(geom.radii)),
            centroids=_point_r2_to_pixel(_t(geom.centroids), H, W),
        )
    if isinstance(geom, PointSet):
        return pixel.PointSet(_point_r2_to_pixel(_t(geom), H, W))
    if isinstance(geom, VectorSet):
        return pixel.VectorSet(_vec_r2_to_pixel(_t(geom)))
    raise NotImplementedError(f"to_pixel does not support {type(geom)}")


def to_normalized(geom, size):
    """Scale r2 (pixel-unit) coordinates by ``2 / min(H, W)`` (isotropic, fill).

    The short axis then spans ``[-1, 1]``; corners on the long axis exceed it. Rotors
    are unitless and unchanged.
    """
    s = min(float(size[0]), float(size[1])) / 2.0
    if isinstance(geom, OrientedBoxSet):
        return OrientedBoxSet(
            radii=_t(geom.radii) / s,
            centroids=_t(geom.centroids) / s,
            rotors=_t(geom.rotors).clone(),
        )
    if isinstance(geom, BoxSet):
        return BoxSet(radii=_t(geom.radii) / s, centroids=_t(geom.centroids) / s)
    if isinstance(geom, (PointSet, VectorSet)):
        return type(geom)(_t(geom) / s)
    raise NotImplementedError(f"to_normalized does not support {type(geom)}")
