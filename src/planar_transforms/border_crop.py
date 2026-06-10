from typing import Any, Callable

from torch import nn

from planar_transforms.functional.border_crop import BordersType, InputType, border_crop


class BorderCrop(nn.Module):
    borders: Any

    def __init__(
        self,
        borders: BordersType | Callable[[], BordersType] | None = None,
    ):
        """
        For convenience borders may be supplied as a Callable.
        This can be used to generate random numbers on the fly.
        """

        super(BorderCrop, self).__init__()
        self.borders = borders

    def forward(self, x: InputType, **kwargs) -> InputType:
        module_kwargs: dict[str, Any] = {}

        if self.borders is not None:
            module_kwargs["borders"] = self.borders() if callable(self.borders) else self.borders

        return border_crop(x, **module_kwargs, **kwargs)
