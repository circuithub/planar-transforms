"""Jointly resample several co-registered fields under one set of parameters.

Co-registered fields (e.g. an image and its label map) must move together under the
same geometry. ``joint_affine`` applies one set of per-sample affine parameters to each
field while honouring its own interpolation contract: continuous fields interpolate,
discrete fields stay nearest. This replaces the dtype-sniffing per-item loop pattern --
the field *type*, not its dtype, decides the resampling.
"""

from typing import Optional

from planar_transforms.functional.affine import ParamType, affine
from planar_transforms.types import ContinuousField, DiscreteField

Field = ContinuousField | DiscreteField


def joint_affine(
    *fields: Field,
    angle: ParamType = 0.0,
    translate: Optional[ParamType] = None,
    scale: ParamType = 1.0,
    shear: Optional[ParamType] = None,
    fill: float = 0.0,
) -> tuple[Field, ...]:
    """Apply one affine transform to every field, per-field interpolation by type.

    All fields must share batch size, height and width. Parameters are CCW degrees /
    pixels / isotropic scale, scalar (shared) or length-B (per sample); see
    :func:`planar_transforms.functional.affine`.
    """
    if not fields:
        return ()
    ref = fields[0]
    for f in fields[1:]:
        assert f.shape[-2:] == ref.shape[-2:], "All fields must share height and width."
        assert f.shape[0] == ref.shape[0], "All fields must share batch size."

    def apply(f: Field) -> Field:
        # Narrow the union so affine()'s field-typed return is preserved per field.
        if isinstance(f, ContinuousField):
            return affine(f, angle=angle, translate=translate, scale=scale, shear=shear, fill=fill)
        return affine(f, angle=angle, translate=translate, scale=scale, shear=shear, fill=fill)

    return tuple(apply(f) for f in fields)
