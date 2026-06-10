"""Tests for planar_transforms resize transform."""

import pytest
import torch

from planar_transforms import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    ImageTensor,
    OrientedBoxSet,
    PointSet,
    Resize,
    VectorSet,
)
from planar_transforms.functional.resize import resize


class TestResizeFunctional:
    """Tests for the functional resize API."""

    def test_continuous_field_upscale(self):
        """Resize a continuous field to larger dimensions."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = resize(x, original_size=(32, 32), new_size=(64, 64))
        assert result.shape == (2, 3, 64, 64)
        assert isinstance(result, ContinuousField)

    def test_continuous_field_downscale(self):
        """Resize a continuous field to smaller dimensions."""
        x = ContinuousField(torch.randn(2, 3, 64, 64))
        result = resize(x, original_size=(64, 64), new_size=(32, 32))
        assert result.shape == (2, 3, 32, 32)

    def test_discrete_field_upscale(self):
        """Resize a discrete field to larger dimensions."""
        # Use float tensor since PyTorch doesn't support nearest upscale for Long
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)).float())
        result = resize(x, original_size=(32, 32), new_size=(64, 64), interpolation="nearest-exact")
        assert result.shape == (2, 1, 64, 64)
        assert isinstance(result, DiscreteField)

    def test_discrete_field_rejects_bilinear(self):
        """Discrete fields should reject non-nearest interpolation."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        with pytest.raises(AssertionError, match="nearest-exact"):
            resize(x, original_size=(32, 32), new_size=(64, 64), interpolation="bilinear")

    def test_discrete_field_rejects_antialias(self):
        """Discrete fields should reject antialiasing."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        with pytest.raises(AssertionError, match="antialiasing"):
            resize(x, original_size=(32, 32), new_size=(64, 64), antialias=True)

    def test_rejects_nearest_interpolation(self):
        """Should reject 'nearest' interpolation (use 'nearest-exact' instead)."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError, match="nearest-exact"):
            resize(x, original_size=(32, 32), new_size=(64, 64), interpolation="nearest")

    def test_image_tensor_bicubic_clamps(self):
        """Bicubic interpolation on ImageTensor should clamp values."""
        # Create an image with values that might overshoot under bicubic
        x = ImageTensor(torch.rand(2, 3, 32, 32))
        result = resize(x, original_size=(32, 32), new_size=(64, 64), interpolation="bicubic")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_vectorset_scaling(self):
        """VectorSet should scale by the resize ratio."""
        x = VectorSet(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
        result = resize(x, original_size=(100, 100), new_size=(200, 200))
        expected = VectorSet(torch.tensor([[20.0, 40.0], [60.0, 80.0]]))
        assert torch.allclose(result, expected)

    def test_pointset_scaling(self):
        """PointSet should scale by the resize ratio."""
        x = PointSet(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
        result = resize(x, original_size=(100, 100), new_size=(50, 100))
        expected = PointSet(torch.tensor([[5.0, 20.0], [15.0, 40.0]]))
        assert torch.allclose(result, expected)

    def test_boxset_scaling(self):
        """BoxSet (radii_x, radii_y, centroid_x, centroid_y) scales per-axis.

        Uses NON-square scaling so a wrong/missing repeat(2) is actually caught:
        a bare ``x * scale_factor`` raises a (4 vs 2) broadcast error.
        """
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        x = BoxSet(radii=radii, centroids=centroids)

        result = resize(x, original_size=(100, 100), new_size=(50, 200))

        # scale = (0.5, 2.0) applied as (sx, sy, sx, sy)
        assert torch.allclose(result.radii, torch.tensor([[2.5, 20.0]]))
        assert torch.allclose(result.centroids, torch.tensor([[10.0, 60.0]]))

    def test_oriented_boxset_scaling(self):
        """OrientedBoxSet should scale radii and centroids but preserve rotors."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        rotors = torch.tensor([[1.0, 0.0]])  # no rotation
        x = OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)

        result = resize(x, original_size=(100, 100), new_size=(200, 200))

        assert torch.allclose(result.radii, torch.tensor([[10.0, 20.0]]))
        assert torch.allclose(result.centroids, torch.tensor([[40.0, 60.0]]))
        assert torch.allclose(result.rotors, rotors)  # rotors unchanged

    def test_size_mismatch_raises(self):
        """Should raise when field dimensions don't match original_size."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError, match="expected"):
            resize(x, original_size=(64, 64), new_size=(128, 128))


class TestResizeModule:
    """Tests for the Resize nn.Module wrapper."""

    def test_module_forward(self):
        """Basic forward pass through Resize module."""
        module = Resize(original_size=(32, 32), new_size=(64, 64))
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 64, 64)

    def test_callable_new_size(self):
        """new_size can be a callable for dynamic sizing."""
        module = Resize(original_size=(32, 32), new_size=lambda: (48, 48))
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 48, 48)

    def test_partial_kwargs(self):
        """Forward can provide missing kwargs."""
        module = Resize(new_size=(64, 64))
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # original_size not set in module, provided at forward time
        result = module(x, original_size=(32, 32))
        assert result.shape == (2, 3, 64, 64)

    def test_interpolation_mode(self):
        """Module should respect interpolation mode."""
        module = Resize(original_size=(32, 32), new_size=(64, 64), interpolation="bicubic")
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 64, 64)

    def test_antialias_option(self):
        """Module should respect antialias option."""
        module = Resize(original_size=(32, 32), new_size=(16, 16), antialias=True)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 16, 16)
