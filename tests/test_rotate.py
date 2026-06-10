"""Tests for planar_transforms rotate transform."""

import pytest
import torch

from planar_transforms import (
    BoxSet,
    ContinuousField,
    Degrees,
    DiscreteField,
    OrientedBoxSet,
    PointSet,
    Rotate,
    VectorSet,
)
from planar_transforms.functional.rotate import rotate


class TestRotateFunctional:
    """Tests for the functional rotate API."""

    def test_continuous_field_90_degrees(self):
        """Rotate a continuous field by 90 degrees."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = rotate(x, angle=Degrees(90.0))
        assert result.shape == (2, 3, 32, 32)
        assert isinstance(result, ContinuousField)

    def test_continuous_field_arbitrary_angle(self):
        """Rotate a continuous field by an arbitrary angle."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = rotate(x, angle=Degrees(45.0))
        assert result.shape == (2, 3, 32, 32)

    def test_discrete_field_uses_nearest(self):
        """Discrete fields should use nearest interpolation by default."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        result = rotate(x, angle=Degrees(90.0))
        assert result.shape == (2, 1, 32, 32)
        assert isinstance(result, DiscreteField)

    def test_discrete_field_rejects_bilinear(self):
        """Discrete fields should reject bilinear interpolation."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        with pytest.raises(AssertionError, match="nearest"):
            rotate(x, angle=Degrees(90.0), interpolation="bilinear")

    def test_vectorset_90_degree_rotation(self):
        """VectorSet rotates to match the image rotation (image-space coords, y down).

        The matrix is transposed so geometry tracks torchvision's image rotation;
        in image coordinates a positive angle sends (1, 0) -> (0, -1).
        See tests/test_image_geometry_alignment.py for the end-to-end proof.
        """
        x = VectorSet(torch.tensor([[1.0, 0.0]]))
        result = rotate(x, angle=Degrees(90.0))
        expected = torch.tensor([[0.0, -1.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_pointset_180_degree_rotation(self):
        """PointSet should rotate points correctly."""
        # (1, 0) rotated 180 degrees -> (-1, 0)
        x = PointSet(torch.tensor([[1.0, 0.0]]))
        result = rotate(x, angle=Degrees(180.0))
        expected = torch.tensor([[-1.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_boxset_rotation(self):
        """BoxSet centroids should rotate while radii stay unchanged."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[10.0, 0.0]])
        x = BoxSet(radii=radii, centroids=centroids)

        result = rotate(x, angle=Degrees(90.0))

        # Radii should be unchanged
        assert torch.allclose(result.radii, radii)
        # Centroids (10, 0) -> (0, -10) at +90 in image-space coords (y down)
        assert torch.allclose(result.centroids, torch.tensor([[0.0, -10.0]]), atol=1e-5)

    def test_oriented_boxset_rotation(self):
        """OrientedBoxSet should rotate centroids and update rotors."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[10.0, 0.0]])
        rotors = torch.tensor([[1.0, 0.0]])  # no initial rotation
        x = OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)

        result = rotate(x, angle=Degrees(90.0))

        # Radii unchanged
        assert torch.allclose(result.radii, radii)
        # Centroids rotated to match the image: (10, 0) -> (0, -10) at +90
        assert torch.allclose(result.centroids, torch.tensor([[0.0, -10.0]]), atol=1e-5)
        # Rotors should now represent 45 degree half-angle (90/2 = 45)
        expected_rotors = torch.tensor(
            [[torch.cos(torch.tensor(torch.pi / 4)), torch.sin(torch.tensor(torch.pi / 4))]]
        )
        assert torch.allclose(result.rotors, expected_rotors, atol=1e-5)

    def test_fit_expand(self):
        """Fit mode 'expand' should expand canvas to fit rotated image."""
        x = ContinuousField(torch.randn(1, 3, 32, 32))
        result = rotate(x, angle=Degrees(45.0), fit="expand")
        # Expanded dimensions should be larger
        assert result.shape[-2] > 32 or result.shape[-1] > 32

    def test_fit_keep_preserves_shape(self):
        """Fit mode 'keep' should preserve original shape."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = rotate(x, angle=Degrees(45.0), fit="keep")
        assert result.shape == x.shape

    def test_center_requires_keep_mode(self):
        """Using center requires fit='keep'."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError, match="not supported with a center"):
            rotate(x, angle=Degrees(45.0), fit="expand", center=torch.tensor([16, 16]))

    def test_batched_different_angles(self):
        """Should handle batch with different rotation angles (image-space)."""
        x = VectorSet(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        angles = torch.tensor([90.0, 180.0])
        result = rotate(x, angle=angles)
        # +90 in image space: (1, 0) -> (0, -1); 180: (1, 0) -> (-1, 0)
        expected = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_zero_rotation(self):
        """Zero rotation should return approximately the same tensor."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = rotate(x, angle=Degrees(0.0))
        assert torch.allclose(result, x, atol=1e-5)


class TestRotateModule:
    """Tests for the Rotate nn.Module wrapper."""

    def test_module_forward(self):
        """Basic forward pass through Rotate module."""
        module = Rotate(angle=90.0)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 32, 32)

    def test_callable_angle(self):
        """angle can be a callable for dynamic angles."""
        module = Rotate(angle=lambda: 45.0)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 32, 32)

    def test_partial_kwargs(self):
        """Forward can provide missing kwargs."""
        module = Rotate(fit="keep")
        x = VectorSet(torch.tensor([[1.0, 0.0]]))
        # angle not set in module, provided at forward time
        result = module(x, angle=Degrees(180.0))
        expected = torch.tensor([[-1.0, 0.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_interpolation_mode(self):
        """Module should respect interpolation mode."""
        module = Rotate(angle=45.0, interpolation="bilinear")
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 32, 32)

    def test_fit_mode(self):
        """Module should respect fit mode."""
        module = Rotate(angle=45.0, fit="keep")
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 32, 32)
