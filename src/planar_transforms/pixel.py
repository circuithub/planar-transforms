"""Pixel-frame geometry.

Image-space coordinates ordered ``(row, col)``, y increasing **down**, origin at the
top-left pixel. This is the storage / IO / rasterization frame.

Convert to the canonical r2 frame (:mod:`planar_transforms.r2`) before applying
geometric transforms; see :func:`planar_transforms.r2.from_pixel` /
:func:`planar_transforms.r2.to_pixel`.
"""

from planar_transforms.types import BoxSet as _BoxSet
from planar_transforms.types import OrientedBoxSet as _OrientedBoxSet
from planar_transforms.types import PointSet as _PointSet
from planar_transforms.types import VectorSet as _VectorSet


class VectorSet(_VectorSet):
    """A displacement in pixel coordinates, ordered ``(drow, dcol)``."""


class PointSet(_PointSet):
    """A position in pixel coordinates, ordered ``(row, col)``; origin top-left."""


class BoxSet(_BoxSet):
    """Axis-aligned box: ``(radii_row, radii_col, centroid_row, centroid_col)``."""


class OrientedBoxSet(_OrientedBoxSet):
    """Oriented box; radii ``(row, col)``, rotor ``(cos, sin)`` half-angle."""
