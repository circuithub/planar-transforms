from typing import Any, Sequence

import torch
from torch import nn


class RandomState:
    """
    A dictionary containing random variables, generated on the fly.
    Variables retain their (original randomized) value unless/until they are explicitly regenerated.
    """

    def __init__(self, generator: torch.Generator = torch.default_generator):
        self.generator = generator
        self.states: dict[str, Any] = {}

    def regenerate(self):
        """
        Regenerates all random variables we've generated so far.
        """
        for v in self.states.values():
            v.regenerate()

    def __getitem__(self, key: str):
        """
        Fetch a previously generated random variable by its key.
        """
        return self.states[key]

    def get(self, key: str, default: Any | None = None):
        """
        Fetch a previously generated random variable by its key, or return a default value if it's not present.
        """
        return self.states.get(key, default)

    def int(
        self,
        key: str,
        *,
        low: int = 0,
        high: int,
    ) -> "RandomInt":
        """
        Generate a random integer in the given range.
        """
        if key in self.states:
            raise ValueError("Key already exists in RandomState")
        val = RandomInt(self.generator, low=low, high=high)
        self.states[key] = val
        return val

    def float(
        self,
        key: str,
        *,
        low: float = 0.0,
        high: float = 1.0,
    ) -> "RandomFloat":
        """
        Generate a random floating point value in the given range.
        """
        if key in self.states:
            raise ValueError("Key already exists in RandomState")
        val = RandomFloat(generator=self.generator, low=low, high=high)
        self.states[key] = val
        return val

    def size(
        self,
        key: str,
        *,
        low: Sequence[int] = (0,),  # type: ignore[valid-type]
        high: Sequence[int],  # type: ignore[valid-type]
    ) -> "RandomSize":
        """
        Generate a random pytorch Size (shape tuple) in the given range.
        """
        if key in self.states:
            raise ValueError("Key already exists in RandomState")
        val = RandomSize(self.generator, low=low, high=high)
        self.states[key] = val
        return val

    def int_tensor(
        self,
        key: str,
        *,
        low: int = 0,  # type: ignore[valid-type]
        high: int,  # type: ignore[valid-type]
        size: tuple,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> "RandomIntTensor":
        """
        Generate a tensor with a specified shape and fill it with random integer values in the given range.
        """
        if key in self.states:
            raise ValueError("Key already exists in RandomState")
        val = RandomIntTensor(
            generator=self.generator,
            low=low,
            high=high,
            size=size,
            device=device,
            dtype=dtype,
        )
        self.states[key] = val
        return val

    def float_tensor(
        self,
        key: str,
        *,
        low: float = 0.0,  # type: ignore[valid-type]
        high: float = 1.0,  # type: ignore[valid-type]
        size: tuple,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> "RandomFloatTensor":
        """
        Generate a tensor with a specified shape and fill it with random floating point values in the given range.
        """
        if key in self.states:
            raise ValueError("Key already exists in RandomState")
        val = RandomFloatTensor(
            generator=self.generator,
            low=low,
            high=high,
            size=size,
            device=device,
            dtype=dtype,
        )
        self.states[key] = val
        return val

    def Regenerate(self) -> "RandomStateRegenerate":
        """
        For use as a PyTorch module. Regenerates all random variables we've generated so far.
        """
        return RandomStateRegenerate(self)


class RandomStateRegenerate(nn.Module):
    def __init__(self, state: RandomState):
        super(RandomStateRegenerate, self).__init__()
        self.state = state

    def forward(self, *args):
        self.state.regenerate()
        if len(args) == 0:
            return
        elif len(args) == 1:
            return args[0]
        else:
            return args

    def __repr__(self):
        return "RandomState.Regenerate()"


class RandomInt:
    def __init__(self, generator, low: int, high: int):
        self.generator = generator
        self.low = low
        self.high = high
        self.val: int | None = None

    def regenerate(self) -> None:
        self.val = torch.randint(self.low, self.high, (1,), generator=self.generator).item()  # type: ignore[assignment]

    def __call__(self) -> int:
        if self.val is None:
            self.regenerate()
        return self.val  # type: ignore[return-value]


class RandomFloat:
    def __init__(self, generator, low: float, high: float):
        self.generator = generator
        self.low = low
        self.high = high
        self.val: float | None = None

    def regenerate(self) -> None:
        self.val = self.low + torch.rand(1, generator=self.generator).item() * (
            self.high - self.low
        )

    def __call__(self) -> float:
        if self.val is None:
            self.regenerate()
        return self.val  # type: ignore[return-value]


class RandomSize:
    def __init__(
        self,
        generator: torch.Generator,
        low: Sequence[int],
        high: Sequence[int],
    ):
        assert len(low) == len(high), "low and high must have the same length"

        self.generator = generator
        self.low = low
        self.high = high
        self.val: torch.Size | None = None

    def regenerate(self) -> None:
        # Fixed: use torch.randint with generator parameter
        self.val = torch.Size(
            [
                int(torch.randint(low, high, (1,), generator=self.generator).item())
                for low, high in zip(self.low, self.high)
            ]
        )

    def __call__(self) -> torch.Size:
        if self.val is None:
            self.regenerate()
        return self.val  # type: ignore[return-value]


class RandomIntTensor:
    def __init__(
        self,
        generator: torch.Generator,
        low: int,
        high: int,
        size: tuple,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self.generator = generator
        self.low = low
        self.high = high
        self.size = size
        self.dtype = dtype
        self.device = device
        self.val: torch.Tensor | None = None

    def regenerate(self) -> None:
        self.val = torch.randint(
            low=self.low,
            high=self.high,
            size=self.size,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )

    def __call__(self) -> torch.Tensor:
        if self.val is None:
            self.regenerate()
        return self.val  # type: ignore[return-value]


class RandomFloatTensor:
    def __init__(
        self,
        generator: torch.Generator,
        low: float,
        high: float,
        size: tuple,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self.generator = generator
        self.low = low
        self.high = high
        self.size = size
        self.dtype = dtype
        self.device = device
        self.val: torch.Tensor | None = None

    def regenerate(self) -> None:
        self.val = self.low + torch.rand(
            *self.size, generator=self.generator, device=self.device, dtype=self.dtype
        ) * (self.high - self.low)

    def __call__(self) -> torch.Tensor:
        if self.val is None:
            self.regenerate()
        return self.val  # type: ignore[return-value]
