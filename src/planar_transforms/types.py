"""
Type definitions for planar transforms.

This module provides tensor subclasses and type aliases used throughout the library.
"""

from typing import NewType

import torch

# NewType with union isn't officially supported, but works fine at runtime.
Degrees = NewType("Degrees", float | torch.Tensor)  # type: ignore[valid-newtype]


class TensorLike(torch.Tensor):
    """
    Treat any subclass of TensorLike similar to a newtype of torch.Tensor.

    In practical terms, this implementation detail lets you support functions like
    torch.where(cond, x, y) with subclassed tensors.

    See https://github.com/docarray/notes/blob/main/blog/02-this-weeks-in-docarray-01.md
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(
            func,
            tuple(torch.Tensor if issubclass(t, TensorLike) else t for t in types),
            args,
            kwargs,
        )


class ContinuousField(TensorLike):
    """
    A tensor representing continuous spatial data that supports interpolation.

    Examples include images where values can be interpolated during resampling.
    Values do not have to be float, but the data should allow for interpolated sampling.
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @property
    def batch_sizes(self) -> torch.Size:
        return self.size()[:-3]  # Assumes shape is B..., C, H, W


class DiscreteField(TensorLike):
    """
    A tensor representing discrete spatial data where interpolation is not valid.

    Examples include indexed segmentation maps. Nearest sampling is the only valid
    sampling on the contained data since values may not be scaled or combined.
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @property
    def batch_sizes(self) -> torch.Size:
        return self.size()[:-3]  # Assumes shape is B..., C, H, W


class ImageTensor(ContinuousField):
    """
    A continuous field representing an image clamped to [0.0, 1.0] or [0, 255].
    """

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        else:
            assert (
                data.is_floating_point() or data.dtype == torch.uint8
            ), "Data type should be float (for range [0.0, 1.0]) or uint8 (for range [0, 255])."
        return torch.Tensor._make_subclass(cls, data, requires_grad)


class VectorSet(TensorLike):
    """A tensor representing a set of vectors."""

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @property
    def batch_sizes(self) -> torch.Size:
        return self.size()[:-1]  # Assumes shape is B..., N


class PointSet(VectorSet):
    """A tensor representing a set of points in space."""

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)


class BoxSet(TensorLike):
    """
    A tensor representing a set of axis-aligned boxes.

    Each box is represented as (radii_x, radii_y, centroid_x, centroid_y).
    """

    def __new__(cls, data=None, radii=None, centroids=None, *args, **kwargs):
        tensors = (radii, centroids)

        if data is not None:
            assert all(t is None for t in tensors)
        elif all(t is None for t in tensors):
            assert data is None
            data = torch.Tensor()
        else:
            assert all(t is not None for t in tensors)
            data = torch.cat(tensors, dim=-1)
        return torch.Tensor._make_subclass(cls, data, *args, **kwargs)

    @property
    def batch_sizes(self) -> torch.Size:
        return self.size()[:-1]  # Assumes shape is B..., 4

    @property
    def radii(self) -> torch.Tensor:
        """Returns a view of the radii of the boxes."""
        return self[..., 0:2].as_subclass(torch.Tensor)

    @property
    def centroids(self) -> torch.Tensor:
        """Returns a view of the centroids of the boxes."""
        return self[..., 2:4].as_subclass(torch.Tensor)


class OrientedBoxSet(BoxSet):
    """
    A tensor representing a set of oriented (rotated) boxes.

    Each box is represented as (radii_x, radii_y, centroid_x, centroid_y, rotor_cos, rotor_sin).
    The rotor is (cos(θ/2), sin(θ/2) * e1∧e2) representing orientation.
    """

    def __new__(cls, data=None, radii=None, centroids=None, rotors=None, *args, **kwargs):
        tensors = (radii, centroids, rotors)

        if data is not None:
            assert all(t is None for t in tensors)
        elif all(t is None for t in tensors):
            assert data is None
            data = torch.Tensor()
        else:
            assert all(t is not None for t in tensors)
            data = torch.cat(tensors, dim=-1)
        return torch.Tensor._make_subclass(cls, data, *args, **kwargs)

    @property
    def batch_sizes(self) -> torch.Size:
        return self.size()[:-1]  # Assumes shape is B..., 6

    @property
    def unoriented(self) -> BoxSet:
        """Returns a view of the unrotated (axis-aligned) boxes."""
        return BoxSet(self[..., :4].as_subclass(torch.Tensor))

    @property
    def rotors(self) -> torch.Tensor:
        """
        Returns a view of the orientation represented as a rotor.

        In 2D this is (cos(θ/2), sin(θ/2) * e1∧e2).
        """
        return self[..., 4:6].as_subclass(torch.Tensor)
