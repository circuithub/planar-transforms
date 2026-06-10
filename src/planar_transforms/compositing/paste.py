import itertools

import torch

from planar_transforms.compositing.intersect_overlay import (
    BlendType,
    InputType,
    OffsetType,
    intersect_overlay,
)
from planar_transforms.types import ContinuousField, DiscreteField


def paste_(
    target: InputType,
    source: InputType,
    offset: OffsetType,
    blend: BlendType,
) -> torch.Tensor:
    """
    Inplace version of paste. Overlays source onto target at the given pixel offset with a blend operation.
    Target, source, and offset will all be broadcast to match each other's batch dimensions.
    """

    device = target.device
    batch_sizes = torch.Size(max(ts, ss) for ts, ss in zip(target.batch_sizes, source.batch_sizes))

    # Convert offset to tensor
    if not isinstance(offset, torch.Tensor):
        offset = torch.tensor(offset, dtype=torch.int, device=device)  # type: ignore[assignment]
    assert isinstance(offset, torch.Tensor)  # type guard for mypy

    assert not offset.dtype.is_floating_point, "Offset must be integer values"

    # Reshape target to (*batch_sizes, ...)
    target = target.expand(*batch_sizes, *target.size()[len(batch_sizes) :])  # type: ignore[assignment]

    # Reshape source to (*batch_sizes, ...)
    source = source.expand(*batch_sizes, *source.size()[len(batch_sizes) :])  # type: ignore[assignment]

    # Reshape offset to (*batch_sizes, 2)
    offset = offset.expand(1) if offset.dim() == 0 else offset
    offset = offset.expand(*batch_sizes, offset.size(dim=-1))

    if offset.size(dim=-1) == 1:
        offset = offset.repeat(*(1 for _ in batch_sizes), 2)
    elif offset.size(dim=-1) != 2:
        raise ValueError(f"Expected offset values of length 1 or 2, but got {offset.size(dim=-1)}")

    if all(isinstance(t, (ContinuousField, DiscreteField)) for t in (target, source)):
        # Ensure that blend is an inplace operation
        def inplace_blend(target, source):
            result = blend(target, source)
            if not target.is_set_to(result):
                target.copy_(result)
            return target

        # Loop over the cartesian product of all batch indices
        for batch_index in itertools.product(*(range(batch_size) for batch_size in batch_sizes)):
            intersect_overlay(  # type: ignore[type-var]
                target=target[batch_index],  # type: ignore[arg-type]
                source=source[batch_index],  # type: ignore[arg-type]
                offset=offset[batch_index],  # type: ignore[arg-type]
                blend=inplace_blend,
            )
        return target  # type: ignore[return-value]
    else:
        raise NotImplementedError(
            f"Unsupported combination of types (target: {type(target)}, source: {type(source)}) for the paste_ transform."
        )


def paste(
    target: InputType,
    source: InputType,
    offset: OffsetType,
    blend: BlendType,
) -> torch.Tensor:
    """
    Overlays source onto target at given offset with a blend operation.
    """
    target = target.clone()  # type: ignore[assignment]
    return paste_(target=target, source=source, offset=offset, blend=blend)
