from typing import Any, Callable, Literal, Optional, Sequence, TypedDict

from torch import nn

from planar_transforms.functional.resize import T, check_resize_kwargs, resize


class ResizeKwargs(TypedDict, total=False):
    original_size: Sequence[int]
    new_size: Sequence[int]
    interpolation: Literal["nearest-exact", "bilinear", "bicubic", "trilinear"]
    antialias: bool


class Resize(nn.Module):
    def __init__(
        self,
        original_size: Optional[Sequence[int]] = None,
        new_size: Optional[Sequence[int] | Callable[[], Sequence[int]]] = None,
        interpolation: Optional[
            Literal["nearest-exact", "bilinear", "bicubic", "trilinear"]
        ] = None,
        antialias: Optional[bool] = None,
    ):
        """
        For convenience new_size may be supplied as a Callable.
        This can be used to generate random numbers on the fly.
        """

        # Early check of kwargs (it will be rechecked in forward)
        check_resize_kwargs(
            original_size=original_size,
            new_size=new_size,
            interpolation=interpolation,
            antialias=antialias,
        )

        super(Resize, self).__init__()

        self.original_size = original_size
        self.new_size = new_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        x: T,
        **kwargs,
    ) -> T:
        module_kwargs: dict[str, Any] = {}

        if self.original_size is not None:
            module_kwargs["original_size"] = self.original_size

        if self.new_size is not None:
            module_kwargs["new_size"] = (
                self.new_size() if callable(self.new_size) else self.new_size
            )

        if self.interpolation is not None:
            module_kwargs["interpolation"] = self.interpolation

        if self.antialias is not None:
            module_kwargs["antialias"] = self.antialias

        return resize(x, **module_kwargs, **kwargs)
