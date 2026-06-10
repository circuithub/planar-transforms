from typing import Any, List, Literal, Optional, TypeVar

import torch
import torchvision

from planar_transforms.types import (
    BoxSet,
    ContinuousField,
    Degrees,
    DiscreteField,
    OrientedBoxSet,
    PointSet,
    VectorSet,
)

T = TypeVar("T", ContinuousField, DiscreteField, VectorSet, PointSet, BoxSet)


def check_rotate_kwargs(
    angle: Optional[Any] = None,
    interpolation: Optional[Literal["nearest", "bilinear"]] = None,
    fit: Optional[Literal["keep", "expand", "shrink"]] = None,
    center: Optional[Any] = None,
    fill: Optional[Any] = None,
):
    if center is not None:
        assert fit == "keep", f"Rotate fit mode {fit} is not supported with a center of rotation."


def vmap_torchvision_rotate(
    batch_images: torch.Tensor,
    batch_angles: Degrees,
    interpolation: Literal["nearest", "bilinear"],
    expand: bool = False,
    batch_centers: Optional[torch.Tensor] = None,
    fill: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Rotate a batch of images with potentially different angles, and centers.

    Pytorch 2's experimental torch.vmap isn't available in the pinned version of pytorch yet, so for now we're just
    using a placeholder shim.
    """
    batch_size = batch_images.shape[0]
    device = batch_images.device

    # Ensure batch_angles is a tensor and expand it if necessary
    if not isinstance(batch_angles, torch.Tensor):
        batch_angles = torch.tensor(batch_angles, device=device)  # type: ignore[assignment]
    batch_angles = batch_angles.expand(batch_size)  # type: ignore[assignment]

    # If batch_centers is provided, ensure it's a tensor and expand it if necessary
    if batch_centers is not None:
        batch_centers = batch_centers.expand(batch_size, -1)

    # Check if we can apply the transformation to the whole batch directly
    if batch_angles.unique_consecutive().size(dim=0) == 1 and (
        batch_centers is None or batch_centers.unique_consecutive(dim=0).size(dim=0) == 1
    ):
        return torchvision.transforms.functional.rotate(  # type: ignore[no-any-return]
            batch_images,
            angle=batch_angles[0].item(),
            interpolation=torchvision.transforms.InterpolationMode(interpolation),
            expand=expand,
            center=batch_centers[0].tolist() if batch_centers is not None else None,
            fill=fill,
        )

    if expand:
        raise ValueError("Expand cannot be used with batched rotation")

    return torch.stack(
        [
            torchvision.transforms.functional.rotate(
                batch_images[i],
                angle=batch_angles[i].item(),
                interpolation=torchvision.transforms.InterpolationMode(interpolation),
                center=batch_centers[i].tolist() if batch_centers is not None else None,
                fill=fill,
            )
            for i in range(batch_size)
        ],
        dim=0,
    )


def vecdot(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Placeholder for pytorch 2's torch.linalg.vecdot which is not available in the current pinned version of pytorch yet.
    """
    return (x.conj() * y).sum(dim=dim)


def rotate(
    x: T,
    angle: Degrees,
    interpolation: Optional[Literal["nearest", "bilinear"]] = None,
    fit: Literal["keep", "expand", "shrink"] = "keep",
    center: Optional[torch.IntTensor] = None,
    fill: Optional[torch.FloatTensor] = None,
) -> T:
    check_rotate_kwargs(angle=angle, interpolation=interpolation, fit=fit, center=center, fill=fill)

    assert center is None or center.shape == (2,), "center must be of shape (2,)"
    assert fill is None or fill.shape == (2,), "fill must be of shape (2,)"

    batch_size = x.size(dim=0)
    device = x.device
    floating_point_dtype = (
        x.dtype
        if x.dtype.is_floating_point and torch.finfo(x.dtype).bits >= torch.finfo(torch.float).bits
        else torch.float
    )
    thetas = (
        (
            angle.to(device)
            if isinstance(angle, torch.Tensor)
            else torch.tensor(angle, dtype=floating_point_dtype, device=device)
        )
        .deg2rad()
        .expand(batch_size)
    )
    if isinstance(x, (ContinuousField, DiscreteField)):
        interpolation = (
            interpolation
            if interpolation is not None
            else "bilinear" if isinstance(x, ContinuousField) else "nearest"
        )
        assert (
            not isinstance(x, DiscreteField) or interpolation == "nearest"
        ), f"Discrete fields cannot use interpolation mode {interpolation}. Please use nearest."

        expand = False
        if fit == "expand":
            expand = True

        def to_dtype(tensor, dtype):
            batch_size, channels, *spatial = tensor.size()
            return (
                torch.view_as_complex(
                    tensor.view(batch_size, channels // 2, 2, *spatial)
                    .permute(0, 1, *range(3, tensor.dim() + 1), 2)
                    .contiguous()
                )
                if dtype.is_complex
                else tensor
            )

        def to_real(tensor):
            batch_size, channels, *spatial = tensor.size()
            return (
                torch.view_as_real(tensor)
                .permute(0, 1, -1, *range(2, tensor.dim()))
                .view(batch_size, channels * 2, *spatial)
                if tensor.is_complex()
                else tensor
            )

        result = to_dtype(
            vmap_torchvision_rotate(
                to_real(x),
                batch_angles=angle,
                interpolation=interpolation,
                expand=expand,
                batch_centers=center,
                fill=fill.tolist() if fill is not None else None,
            ),
            dtype=x.dtype,
        )

        if fit == "shrink":
            raise NotImplementedError("Fit mode shrink is untested at this time")

        assert (
            fit != "keep" or result.shape == x.shape
        ), f"Postcondition violated: {result.shape} does not match {x.shape}"
        return result  # type: ignore[return-value, no-any-return]
    elif isinstance(x, (VectorSet, PointSet, BoxSet, OrientedBoxSet)):
        if center is not None:
            raise NotImplementedError("Rotation center is not supported for sets at this time")

        # 2d rotation matrix, transposed so geometry rotates the SAME visual
        # direction as the (torchvision) image rotation. The transpose is required
        # because point/centroid coords are image-space (col, row) with y pointing
        # DOWN; it maps math-convention CCW onto image space. Do not "simplify" it
        # away — see tests/test_image_geometry_alignment.py.
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        rotation_matrices = torch.stack(
            [
                torch.stack([cos_thetas, -sin_thetas], dim=-1),
                torch.stack([sin_thetas, cos_thetas], dim=-1),
            ],
            dim=-2,
        ).transpose(-2, -1)

        for _ in range(x.dim() - 2):
            # Handle vector shapes with (B, *, N) where * indicates extra dimensions
            rotation_matrices = rotation_matrices[:, None, ...]

        # Apply rotations
        if isinstance(x, (VectorSet, PointSet)):
            # Batched matrix-vector multiplication (B,M=2,N=2) × (B,...,N=2) -> B,...,M
            return vecdot(rotation_matrices, x[..., None, :], dim=-1)  # type: ignore[return-value]
        elif isinstance(x, OrientedBoxSet):
            rotation_rotor = torch.stack([torch.cos(thetas / 2), torch.sin(thetas / 2)], dim=-1)
            return OrientedBoxSet(
                radii=x.radii,
                centroids=(
                    # Batched matrix-vector multiplication (B,M=2,N=2) × (B,...,N=2) -> B,...,M
                    vecdot(rotation_matrices, x.centroids[..., None, :], dim=-1)
                ),
                rotors=torch.view_as_real(
                    # 2D rotors are essentially complex rotations (with half angles)
                    torch.view_as_complex(rotation_rotor)
                    * torch.view_as_complex(x.rotors)
                ),
            )
        elif isinstance(x, BoxSet):
            return BoxSet(
                radii=x.radii,
                centroids=(
                    # Batched matrix-vector multiplication (B,M=2,N=2) × (B,...,N=2) -> B,...,M
                    vecdot(rotation_matrices, x.centroids[..., None, :], dim=-1)
                ),
            )
        else:
            assert False, "Unhandled type"
    elif x.is_complex():
        # Simple complex number set
        complexrotation = torch.cos(thetas) + torch.sin(thetas) * 1j
        return x * complexrotation  # type: ignore[return-value]
    else:
        raise NotImplementedError(f"Unsupported input type {type(x)} for the rotate transform.")
