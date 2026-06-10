from typing import Callable, Sequence, TypeAlias, TypeVar

import torch

from planar_transforms.types import (
    ContinuousField,
    DiscreteField,
)

InputType = TypeVar("InputType", ContinuousField, DiscreteField)
OffsetType = TypeVar("OffsetType", int, Sequence[int], torch.Tensor)
BlendType: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def intersect_overlay(
    target: InputType,
    source: InputType,
    offset: OffsetType,
    blend: BlendType,
) -> torch.Tensor:
    """
    Overlay source image onto target image at a given offset with a specified blend operation.
    The operation can be inplace or not depending on the blend function.

    Returns the intersected region (even if blend is an inplace operation).
    """

    device = target.device
    batch_sizes = target.batch_sizes

    # Cannot easily support batches because we want to support inplace application of intersect_overlay.
    # It's possible to support it for uniform offsets in future, similar to pad, but not for inplace blend since it's
    # not possible to stack views into larger views, even if they're the same shape.
    if sum(batch_sizes) not in (0, 1):
        raise NotImplementedError(
            "Batch implementation of intersect_overlay is not supported at this time. Consider using paste or paste_."
        )

    # Convert offset to tensor
    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor(offset, dtype=torch.int, device=device)  # type: ignore[assignment]
    assert isinstance(offset, torch.Tensor)  # type guard for mypy

    assert not offset.dtype.is_floating_point, "Offset must be integer values"

    # Reshape offset to (*batch_sizes, 2)
    offset = offset.expand(1) if offset.dim() == 0 else offset

    if offset.size() == (1,):
        offset = offset.repeat(2)
    elif offset.size() != (2,):
        raise ValueError(f"Expected offset values of length 1 or 2, but got {offset.size(dim=-1)}")

    if all(isinstance(t, (ContinuousField, DiscreteField)) for t in (target, source)):
        if target.dim() != source.dim():
            raise ValueError(
                f"Target and source tensors must have the same number of dimensions, got (target.dim={source.dim()}, source.dim={target.dim()})"
            )

        # Separate positive and negative offsets for clipping calculations
        x, y = offset
        xpos, ypos = offset.relu()
        xneg, yneg = (-offset).relu()

        # Clip source view on left and top borders (if negative position is supplied)
        source = source[..., yneg:, xneg:]  # type: ignore[assignment]

        # Clip target view to paste region
        target = target[  # type: ignore[assignment]
            ...,
            # Note that slicing automatically clamps to target's max indices
            ypos : max(0, y + source.size(dim=-2)),
            xpos : max(0, x + source.size(dim=-1)),
        ]

        # The fully clipped region is the size of the target view
        clipped_height, clipped_width = target.size()[-2:]

        # Clip source to the paste region
        source = source[..., yneg : (yneg + clipped_height), xneg : (xneg + clipped_width)]  # type: ignore[assignment]

        # Blend the source and target using the blending function
        result = blend(target, source)

        # This may or may not be a view of the supplied target tensor depending on whether or not blend is inplace
        return result
    else:
        raise NotImplementedError(
            f"Unsupported combination of types (target: {type(target)}, source: {type(source)}) for the intersect_overlay transform."
        )
