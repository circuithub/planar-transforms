from typing import Any, Callable

from torch import nn

from planar_transforms.functional.pad import InputType, PadType, pad


class Pad(nn.Module):
    pad: Any
    mode: str | None
    value: float | None

    def __init__(
        self,
        pad: PadType | Callable[[], PadType] | None = None,
        mode: str | None = None,
        value: float | None = None,
    ):
        """
        For convenience pad may be supplied as a Callable.
        This can be used to generate random numbers on the fly.
        """

        super(Pad, self).__init__()

        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, x: InputType, **kwargs) -> InputType:
        module_kwargs: dict[str, Any] = {}

        if self.pad is not None:
            module_kwargs["pad"] = self.pad() if callable(self.pad) else self.pad

        if self.mode is not None:
            module_kwargs["mode"] = self.mode

        if self.value is not None:
            module_kwargs["value"] = self.value

        return pad(x, **module_kwargs, **kwargs)
