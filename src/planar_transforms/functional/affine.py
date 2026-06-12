"""Batched affine resampling of continuous and discrete fields.

A single ``grid_sample`` call applies per-sample affine parameters to a whole batch
on the GPU, replacing the per-item resampling loop. Angles follow the library's
counter-clockwise r2 convention (see :mod:`planar_transforms.r2`), so a positive
``angle`` here rotates a field the same way :func:`planar_transforms.functional.rotate`
does.
"""

from typing import Literal, Optional, Sequence, TypeVar

import torch
import torch.nn.functional as F

from planar_transforms.types import ContinuousField, DiscreteField

T = TypeVar("T", ContinuousField, DiscreteField)

# Per-sample affine parameters. Scalars broadcast across the batch; a length-B or
# (B, ...) tensor gives one value per sample. Angles/shears are in degrees, CCW.
ParamType = float | Sequence[float] | torch.Tensor


def _as_batch(value: ParamType, batch_size: int, device: torch.device) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    t = t.to(device=device, dtype=torch.float)
    return t.expand(batch_size) if t.dim() == 0 else t


def affine_theta(
    *,
    angle: torch.Tensor,
    translate: torch.Tensor,
    scale: torch.Tensor,
    shear: torch.Tensor,
    size: torch.Size,
) -> torch.Tensor:
    """Build per-sample ``grid_sample`` theta from affine parameters.

    Returns a ``(B, 2, 3)`` tensor mapping normalized output coordinates to normalized
    input coordinates (the inverse map ``grid_sample`` expects), with ``align_corners=True``
    pixel<->normalized scaling. Pure-rotation theta matches
    :func:`planar_transforms.functional.rotate` exactly.

    ``angle``/``shear`` are CCW degrees; ``translate`` is ``(B, 2)`` in pixels ``(tx, ty)``
    with y up; ``scale`` is ``(B,)`` isotropic. ``size`` is ``(H, W)``.
    """
    device = angle.device
    H, W = int(size[0]), int(size[1])
    rot = torch.deg2rad(angle)
    # CCW convention: negate to match torchvision's CW-positive affine matrix builder.
    sx = torch.deg2rad(-shear[..., 0])
    sy = torch.deg2rad(-shear[..., 1])
    nrot = -rot

    cos_sy = torch.cos(sy)
    a = torch.cos(nrot - sy) / cos_sy
    b = -torch.cos(nrot - sy) * torch.tan(sx) / cos_sy - torch.sin(nrot)
    c = torch.sin(nrot - sy) / cos_sy
    d = -torch.sin(nrot - sy) * torch.tan(sx) / cos_sy + torch.cos(nrot)

    # Inverse (output->input) pixel-centered linear map, with isotropic scale.
    m00, m01 = d / scale, -b / scale
    m10, m11 = -c / scale, a / scale

    # Inverse translation in pixels: output->input subtracts the forward translate.
    # Pixel y is down, r2/translate y is up, so the row (y) translate flips sign.
    tx = -translate[..., 0]
    ty = translate[..., 1]
    m02 = m00 * tx + m01 * ty
    m12 = m10 * tx + m11 * ty

    # Compose pixel<->normalized scaling: N = Sinv @ M @ S with S = diag((W-1)/2, (H-1)/2).
    sxn, syn = (W - 1) / 2.0, (H - 1) / 2.0
    theta = torch.zeros(angle.shape[0], 2, 3, device=device)
    theta[:, 0, 0] = m00
    theta[:, 0, 1] = m01 * syn / sxn
    theta[:, 0, 2] = m02 / sxn
    theta[:, 1, 0] = m10 * sxn / syn
    theta[:, 1, 1] = m11
    theta[:, 1, 2] = m12 / syn
    return theta


def affine(
    x: T,
    *,
    angle: ParamType = 0.0,
    translate: Optional[ParamType] = None,
    scale: ParamType = 1.0,
    shear: Optional[ParamType] = None,
    interpolation: Optional[Literal["nearest", "bilinear"]] = None,
    fill: float = 0.0,
) -> T:
    """Apply a per-sample affine transform to a batched field via one ``grid_sample`` call.

    ``angle`` and ``shear`` are CCW degrees (``shear`` is ``(sx, sy)``), ``translate`` is
    pixels ``(tx, ty)`` with y up, ``scale`` is isotropic. Each may be a scalar (shared)
    or a length-B tensor (per sample). The field type sets the interpolation contract:
    continuous defaults to bilinear, discrete to nearest and rejects bilinear.
    """
    if not isinstance(x, (ContinuousField, DiscreteField)):
        raise NotImplementedError(f"Unsupported input type {type(x)} for the affine transform.")

    interpolation = (
        interpolation
        if interpolation is not None
        else "bilinear" if isinstance(x, ContinuousField) else "nearest"
    )
    assert (
        not isinstance(x, DiscreteField) or interpolation == "nearest"
    ), f"Discrete fields cannot use interpolation mode {interpolation}. Please use nearest."

    batched = x if x.dim() == 4 else x[None, ...]
    B, _, H, W = batched.shape
    device = batched.device
    size = torch.Size((H, W))

    angle_t = _as_batch(angle, B, device)
    scale_t = _as_batch(scale, B, device)
    if translate is None:
        translate_t = torch.zeros(B, 2, device=device)
    else:
        translate_t = (
            translate
            if isinstance(translate, torch.Tensor)
            else torch.tensor(translate, device=device)
        )
        translate_t = translate_t.to(device=device, dtype=torch.float).expand(B, 2)
    if shear is None:
        shear_t = torch.zeros(B, 2, device=device)
    else:
        shear_t = shear if isinstance(shear, torch.Tensor) else torch.tensor(shear, device=device)
        shear_t = shear_t.to(device=device, dtype=torch.float).expand(B, 2)

    theta = affine_theta(
        angle=angle_t, translate=translate_t, scale=scale_t, shear=shear_t, size=size
    )

    # grid_sample needs a float field; remember the original dtype to restore it.
    float_field = batched.to(torch.float)
    grid = F.affine_grid(theta, [B, batched.shape[1], H, W], align_corners=True)
    sampled = (
        F.grid_sample(
            float_field - fill,
            grid,
            mode=interpolation,
            padding_mode="zeros",
            align_corners=True,
        )
        + fill
    )

    result = sampled.to(batched.dtype)
    result = result if x.dim() == 4 else result[0]
    return type(x)(result)
