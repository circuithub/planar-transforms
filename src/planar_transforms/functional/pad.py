from typing import Sequence, TypeVar

import torch
import torch.nn as nn

from planar_transforms.types import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    OrientedBoxSet,
    PointSet,
    VectorSet,
)

InputType = TypeVar("InputType", ContinuousField, DiscreteField, VectorSet, PointSet, BoxSet)
PadType = TypeVar("PadType", int, Sequence[int], torch.Tensor)


def pad(
    x: InputType,
    pad: PadType,
    mode: str = "constant",
    value: float | None = None,
) -> InputType:
    """Pads or crops a tensor image/field depending on the padding values specified:
    * For positive padding values, the tensor is padded.
    * For negative padding values, the tensor is cropped.

    In vector spaces, the corresponding translation of the origin is applied instead. This approach allows for operations in a coordinate system that is independent of the image's origin.

    When positive padding is used we delegate to torch.nn.functional.pad and therefore support mode and value arguments.
    """

    device = x.device
    batch_sizes = x.batch_sizes

    # Convert pad to tensor
    if not isinstance(pad, torch.Tensor):
        pad = torch.tensor(pad, dtype=torch.int, device=device)  # type: ignore[assignment]
    assert isinstance(pad, torch.Tensor)  # type guard for mypy

    assert not pad.dtype.is_floating_point, "padding must be integer values"

    # Reshape pad to (*batch_sizes, 4)
    pad = pad.expand(1) if pad.dim() == 0 else pad
    pad = pad.expand(*x.batch_sizes, pad.size(dim=-1))

    if pad.size(dim=-1) == 1:
        pad = pad.repeat(*(1 for _ in batch_sizes), 4)
    elif pad.size(dim=-1) == 2:
        pad = pad.repeat(*(1 for _ in batch_sizes), 2)
    elif pad.size(dim=-1) != 4:
        raise ValueError(f"Expected pad values of length 1, 2, or 4, but got {pad.size(dim=-1)}")

    # Check for uniform output dimension across the batch
    total_pad = pad[..., [0, 2]] + pad[..., [1, 3]]
    if total_pad.dim() > 1 and total_pad.unique_consecutive(dim=0).size(dim=0) != 1:
        raise ValueError(
            f"""Padding/cropping must result in uniform dimensions for all images in the batch, but got
{pad[:, [0,2]]} +
{pad[:, [1,3]]} =
{total_pad}"""
        )

    if isinstance(x, (ContinuousField, DiscreteField)):
        # Flatten batch dimensions to B
        x = x.flatten(end_dim=len(batch_sizes) - 1) if batch_sizes != () else x[None, ...]  # type: ignore[assignment]

        pad = pad.flatten(end_dim=-2) if batch_sizes != () else pad[None, ...]

        # Separate crop and pad arguments
        crop = (-pad).relu()
        pad = pad.relu()

        # Image bounds checking
        crop_width, crop_height = (
            x.size(dim=-1) - (crop[:, 0] + crop[:, 1]),  # width
            x.size(dim=-2) - (crop[:, 2] + crop[:, 3]),  # height
        )
        if (crop_width < 0).any() or (crop_height < 0).any():
            raise ValueError(
                f"Negative padding results in negative dimensions {(crop_width, crop_height)}"
            )

        # Crop each batch element
        crop_unique = crop.unique_consecutive(dim=0)
        if crop_unique.size(dim=0) == 1:
            crop_left, crop_right, crop_top, crop_bottom = crop_unique.squeeze(dim=0).unbind(dim=-1)
            cropped = [
                x[
                    :,  # B
                    :,  # C
                    crop_top : (-crop_bottom if crop_bottom > 0 else x.size(dim=-2)),  # H
                    crop_left : (-crop_right if crop_right > 0 else x.size(dim=-1)),  # W
                ]
            ]
        else:
            cropped = [
                x_slice[
                    None,  # B
                    :,  # C
                    crop_top : (-crop_bottom if crop_bottom > 0 else x.size(dim=-2)),  # H
                    crop_left : (-crop_right if crop_right > 0 else x.size(dim=-1)),  # W
                ]
                for x_slice, crop_left, crop_right, crop_top, crop_bottom in zip(
                    x, *crop.unbind(dim=-1)
                )
            ]

        # Check if padding is required
        if (pad == 0).all():
            cropped_t = torch.cat(cropped, dim=0)
            if batch_sizes == ():
                return cropped_t.squeeze(dim=0)  # type: ignore[return-value, no-any-return]
            else:
                return cropped_t.unflatten(dim=0, sizes=batch_sizes)  # type: ignore[return-value, no-any-return]

        # Pad each batch element
        pad_unique = pad.unique_consecutive(dim=0)
        if pad_unique.size(dim=0) == 1 and len(cropped) == 1:
            pad_left, pad_right, pad_top, pad_bottom = pad_unique.squeeze(dim=0).unbind(dim=-1)
            padded = [
                nn.functional.pad(
                    cropped[0],
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode=mode,
                    value=value,
                )
            ]
        else:
            padded = [
                nn.functional.pad(
                    cropped_slice[None],
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode=mode,
                    value=value,
                )
                for cropped_slice, pad_left, pad_right, pad_top, pad_bottom in zip(
                    cropped, *pad.unbind(dim=-1)
                )
            ]
        padded_t = torch.cat(padded, dim=0)
        if batch_sizes == ():
            return padded_t.squeeze(dim=0)  # type: ignore[return-value, no-any-return]
        else:
            return padded_t.unflatten(dim=0, sizes=batch_sizes)  # type: ignore[return-value, no-any-return]

    elif isinstance(x, (VectorSet, PointSet, BoxSet, OrientedBoxSet)):
        translation = (
            # (left - right, bottom - top) / 2
            (pad[..., [0, 3]] - pad[..., [1, 2]]).to(dtype=x.dtype)
            * 0.5
        )
        if isinstance(x, (VectorSet, PointSet)):
            return x + translation  # type: ignore[return-value, no-any-return]
        if isinstance(x, OrientedBoxSet):
            return OrientedBoxSet(torch.cat((x.radii, x.centroids + translation, x.rotors), dim=-1))
        elif isinstance(x, BoxSet):
            return BoxSet(torch.cat((x.radii, x.centroids + translation), dim=-1))
        else:
            assert False, "Unhandled type"
    else:
        raise NotImplementedError(f"Unsupported input type {type(x)} for the pad transform.")
