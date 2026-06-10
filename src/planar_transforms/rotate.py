from typing import Any, Callable, Literal, Optional, TypedDict

import torch
from torch import nn

from planar_transforms.functional.rotate import T, check_rotate_kwargs, rotate
from planar_transforms.types import Degrees


class RotateKwargs(TypedDict, total=False):
    angle: Degrees
    interpolation: Literal["nearest", "bilinear"]
    fit: Literal["keep", "expand", "shrink"]
    center: Optional[torch.IntTensor]
    fill: Optional[torch.FloatTensor]


class Rotate(nn.Module):
    def __init__(
        self,
        angle: Optional[float | Callable[[], float]] = None,
        interpolation: Optional[Literal["nearest", "bilinear"]] = None,
        fit: Optional[Literal["keep", "expand", "shrink"]] = None,
        center: Optional[torch.IntTensor | Callable[[], torch.IntTensor]] = None,
        fill: Optional[torch.FloatTensor | Callable[[], torch.FloatTensor]] = None,
    ):
        """
        For convenience angle, center and fill may be supplied as Callables.
        This can be used to generate random numbers for these parameters on the fly.
        """

        # Early check of kwargs (it will be rechecked in forward)
        check_rotate_kwargs(
            angle=angle, interpolation=interpolation, fit=fit, center=center, fill=fill
        )
        super(Rotate, self).__init__()

        self.angle = angle
        self.interpolation = interpolation
        self.fit = fit
        self.center = center
        self.fill = fill

    def forward(
        self,
        x: T,
        **kwargs,
    ) -> T:
        module_kwargs: dict[str, Any] = {}

        if self.angle is not None:
            module_kwargs["angle"] = Degrees(self.angle() if callable(self.angle) else self.angle)

        if self.interpolation is not None:
            module_kwargs["interpolation"] = self.interpolation

        if self.fit is not None:
            module_kwargs["fit"] = self.fit

        if self.center is not None:
            module_kwargs["center"] = self.center() if callable(self.center) else self.center

        if self.fill is not None:
            module_kwargs["fill"] = self.fill() if callable(self.fill) else self.fill

        return rotate(x, **module_kwargs, **kwargs)
