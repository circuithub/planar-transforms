from typing import Any, Callable, Literal, Optional

from torch import nn

from planar_transforms.functional.affine import ParamType, T, affine


class Affine(nn.Module):
    """Module wrapper for :func:`planar_transforms.functional.affine`.

    Each parameter may be a value or a zero-arg callable (e.g. a random generator)
    evaluated per forward call, matching the :class:`planar_transforms.Rotate` idiom.
    """

    def __init__(
        self,
        angle: Optional[ParamType | Callable[[], ParamType]] = None,
        translate: Optional[ParamType | Callable[[], ParamType]] = None,
        scale: Optional[ParamType | Callable[[], ParamType]] = None,
        shear: Optional[ParamType | Callable[[], ParamType]] = None,
        interpolation: Optional[Literal["nearest", "bilinear"]] = None,
        fill: Optional[float] = None,
    ):
        super(Affine, self).__init__()
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, x: T, **kwargs) -> T:
        module_kwargs: dict[str, Any] = {}
        for name in ("angle", "translate", "scale", "shear"):
            value = getattr(self, name)
            if value is not None:
                module_kwargs[name] = value() if callable(value) else value
        if self.interpolation is not None:
            module_kwargs["interpolation"] = self.interpolation
        if self.fill is not None:
            module_kwargs["fill"] = self.fill
        return affine(x, **module_kwargs, **kwargs)
