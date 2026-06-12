from typing import Any, Callable, Optional

import torch
from torch import nn

from planar_transforms.functional.reflect import Axis, T, reflect


class Reflect(nn.Module):
    """Module wrapper for :func:`planar_transforms.functional.reflect`.

    ``axis`` and ``mask`` may be values or zero-arg callables evaluated per forward call
    (e.g. a random per-sample mask), matching the :class:`planar_transforms.Rotate` idiom.
    """

    def __init__(
        self,
        axis: Optional[Axis | Callable[[], Axis]] = None,
        mask: Optional[torch.Tensor | Callable[[], torch.Tensor]] = None,
    ):
        super(Reflect, self).__init__()
        self.axis = axis
        self.mask = mask

    def forward(self, x: T, **kwargs) -> T:
        module_kwargs: dict[str, Any] = {}
        if self.axis is not None:
            module_kwargs["axis"] = self.axis() if callable(self.axis) else self.axis
        if self.mask is not None:
            module_kwargs["mask"] = self.mask() if callable(self.mask) else self.mask
        return reflect(x, **module_kwargs, **kwargs)
