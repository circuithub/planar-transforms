from typing import Sequence, TypeVar

import torch

from planar_transforms.functional.pad import InputType, pad

BordersType = TypeVar("BordersType", int, Sequence[int], torch.Tensor)


def border_crop(
    x: InputType,
    borders: BordersType,
) -> InputType:
    """Differs from torchvision crop in that crop amounts is subtracted from the borders. This allows us to apply crop as
    a translation (e.g. to bounding boxes) without knowing the image dimensions.

    This is implemented using negative padding (see transforms.functional.pad)"""
    # Convert borders to tensor
    if not isinstance(borders, torch.Tensor):
        borders = torch.tensor(borders, dtype=torch.int, device=x.device)  # type: ignore[assignment]
    assert isinstance(borders, torch.Tensor)  # type guard for mypy

    assert not borders.dtype.is_floating_point, "borders must be integer values"

    # Check for valid borders values
    if borders.size()[-1:] not in ((), (1,), (2,), (4,)):
        raise ValueError(
            f"Expected a borders values of length 1, 2, or 4, but got {borders.size()}"
        )
    if (borders < 0).any():
        raise ValueError(f"Expected all positive borders values, but got {borders}")

    return pad(x, pad=-borders)
