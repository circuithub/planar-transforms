from typing import Any, Literal, Optional, Sequence, TypeVar

import torch
import torch.nn as nn

from planar_transforms.types import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    ImageTensor,
    OrientedBoxSet,
    PointSet,
    VectorSet,
)

T = TypeVar("T", ContinuousField, DiscreteField, VectorSet, PointSet, BoxSet)


def check_resize_kwargs(
    original_size: Optional[Sequence[int]] = None,
    new_size: Optional[Any] = None,
    interpolation: Optional[Literal["nearest-exact", "bilinear", "bicubic", "trilinear"]] = None,
    antialias: Optional[bool] = None,
):
    assert (
        interpolation != "nearest"
    ), "Interpolation mode nearest is not recommended because it truncates sampling indices, which doesn't align well with other interpolation modes. Use nearest-exact instead."


def resize(
    x: T,
    original_size: Sequence[int],
    new_size: Sequence[int],
    interpolation: Optional[Literal["nearest-exact", "bilinear", "bicubic", "trilinear"]] = None,
    antialias: Optional[bool] = None,
) -> T:
    check_resize_kwargs(
        original_size=original_size,
        new_size=new_size,
        interpolation=interpolation,
        antialias=antialias,
    )

    assert len(original_size) == 2, "original_size must have length 2"
    assert len(new_size) == 2, "new_size must have length 2"
    original_size = torch.Size(original_size)
    new_size = torch.Size(new_size)

    device = x.device
    if isinstance(x, (ContinuousField, DiscreteField)):
        assert (
            x.shape[-2:] == original_size
        ), f"Field {type(x)} height and width is {x.shape[-2:]}, expected {original_size}"
        interpolation = (
            interpolation
            if interpolation is not None
            else "bilinear" if isinstance(x, ContinuousField) else "nearest-exact"
        )
        assert (
            not isinstance(x, DiscreteField) or interpolation == "nearest-exact"
        ), f"Discrete fields cannot use interpolation mode {interpolation}. Please use nearest-exact."
        antialias = (
            antialias
            if antialias is not None
            else True if isinstance(x, ContinuousField) else False
        )
        assert (
            not isinstance(x, DiscreteField) or antialias is False
        ), f"Discrete fields cannot use antialiasing (set to {antialias})."
        result = nn.functional.interpolate(
            x,
            size=new_size,
            mode=interpolation,
            align_corners=None,
            antialias=antialias,
        )
        if isinstance(x, ImageTensor) and interpolation == "bicubic":
            # Clamp overshoot that can happen as a result of bicubic sampling
            if result.is_floating_point():
                result = result.clamp(min=0.0, max=1.0)
            elif result.is_signed():
                result = result.clamp(min=0, max=255)
            elif result.dtype != torch.uint8:
                result = result.minimum(torch.tensor(255, device=device))
        assert (
            result.shape[-2:] == new_size
        ), f"Unexpected output height and width {result.shape[-2:]}, expected {new_size}."
        return result  # type: ignore[return-value]
    elif isinstance(x, (VectorSet, PointSet, BoxSet, OrientedBoxSet)):
        scale_factor = torch.tensor(
            [
                new_size[0] / original_size[0],
                new_size[1] / original_size[1],
            ],
            device=device,
            dtype=(
                x.dtype
                if x.dtype.is_floating_point
                and torch.finfo(x.dtype).bits >= torch.finfo(torch.float).bits
                else torch.float
            ),
        )
        if isinstance(x, (VectorSet, PointSet)):
            return x * scale_factor  # type: ignore[return-value]
        elif isinstance(x, OrientedBoxSet):
            return OrientedBoxSet(
                torch.cat((x.unoriented * scale_factor.repeat(2), x.rotors), dim=-1)
            )
        elif isinstance(x, BoxSet):
            # BoxSet is (radii_x, radii_y, centroid_x, centroid_y) -> repeat the
            # 2-element scale to (sx, sy, sx, sy). Without repeat(2) this raises a
            # (4 vs 2) broadcast error. See tests/test_resize.py::test_boxset_scaling.
            return x * scale_factor.repeat(2)  # type: ignore[return-value]
        else:
            assert False, "Unhandled type"
    else:
        raise NotImplementedError(f"Unsupported input type {type(x)} for the resize transform.")
