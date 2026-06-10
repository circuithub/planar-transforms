"""Tests for planar_transforms border_crop transform."""

import pytest
import torch

from planar_transforms import (
    BorderCrop,
    BoxSet,
    ContinuousField,
    DiscreteField,
    OrientedBoxSet,
    PointSet,
    VectorSet,
)
from planar_transforms.functional.border_crop import border_crop


class TestBorderCropFunctional:
    """Tests for the functional border_crop API."""

    def test_continuous_field_uniform_crop(self):
        """Crop a continuous field uniformly on all sides."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = border_crop(x, borders=4)
        assert result.shape == (2, 3, 24, 24)
        assert isinstance(result, ContinuousField)

    def test_continuous_field_asymmetric_crop(self):
        """Crop a continuous field with different amounts per side."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # (left, right, top, bottom)
        result = border_crop(x, borders=[1, 2, 3, 4])
        assert result.shape == (2, 3, 25, 29)  # H: 32-3-4=25, W: 32-1-2=29

    def test_discrete_field_crop(self):
        """Crop a discrete field."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        result = border_crop(x, borders=2)
        assert result.shape == (2, 1, 28, 28)
        assert isinstance(result, DiscreteField)

    def test_vectorset_translation(self):
        """VectorSet should translate (opposite direction to padding)."""
        x = VectorSet(torch.tensor([[10.0, 20.0]]))
        # Cropping left=4, right=0 means origin shifts right by 2
        # borders -> -pad, so translation is -(4-0)/2 = -2
        result = border_crop(x, borders=[4, 0, 0, 2])
        expected = torch.tensor([[8.0, 19.0]])
        assert torch.allclose(result, expected)

    def test_pointset_symmetric_crop(self):
        """PointSet with symmetric crop should not translate."""
        x = PointSet(torch.tensor([[16.0, 16.0]]))
        result = border_crop(x, borders=4)
        assert torch.allclose(result, x)

    def test_boxset_translation(self):
        """BoxSet centroids should translate while radii stay unchanged."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        x = BoxSet(radii=radii, centroids=centroids)

        # borders -> negative pad -> opposite translation
        result = border_crop(x, borders=[10, 0, 0, 6])

        assert torch.allclose(result.radii, radii)
        # Translation should be opposite: (-5, -3)
        assert torch.allclose(result.centroids, torch.tensor([[15.0, 27.0]]))

    def test_oriented_boxset_translation(self):
        """OrientedBoxSet centroids should translate while radii and rotors stay unchanged."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        rotors = torch.tensor([[1.0, 0.0]])
        x = OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)

        result = border_crop(x, borders=[10, 0, 0, 6])

        assert torch.allclose(result.radii, radii)
        assert torch.allclose(result.centroids, torch.tensor([[15.0, 27.0]]))
        assert torch.allclose(result.rotors, rotors)

    def test_tensor_border_values(self):
        """Borders can be specified as a tensor."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        borders_tensor = torch.tensor([1, 2, 3, 4])
        result = border_crop(x, borders=borders_tensor)
        assert result.shape == (2, 3, 25, 29)

    def test_float_borders_raises(self):
        """Floating point border values should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError, match="integer"):
            border_crop(x, borders=torch.tensor([1.5, 2.0, 3.0, 4.0]))

    def test_negative_borders_raises(self):
        """Negative border values should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError, match="positive"):
            border_crop(x, borders=-4)

    def test_excessive_crop_raises(self):
        """Cropping more than image size should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError, match="negative dimensions"):
            border_crop(x, borders=20)

    def test_invalid_borders_length_raises(self):
        """Invalid borders length should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError, match="1, 2, or 4"):
            border_crop(x, borders=[1, 2, 3])

    def test_batch_dimensions(self):
        """Border crop should preserve batch dimensions."""
        x = ContinuousField(torch.randn(4, 2, 3, 32, 32))
        result = border_crop(x, borders=2)
        assert result.shape == (4, 2, 3, 28, 28)

    def test_two_value_borders(self):
        """Borders with 2 values should apply to (left/right, top/bottom)."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = border_crop(x, borders=[2, 4])
        # borders=[2, 4] expands to [2, 4, 2, 4] -> H: 32-2-4=26, W: 32-2-4=26
        # Wait, let me check the pad expansion logic again...
        # In pad.py: if length 2, repeat twice -> [2, 4, 2, 4]
        # That means: left=2, right=4, top=2, bottom=4
        # H: 32-2-4=26, W: 32-2-4=26
        assert result.shape == (2, 3, 26, 26)


class TestBorderCropModule:
    """Tests for the BorderCrop nn.Module wrapper."""

    def test_module_forward(self):
        """Basic forward pass through BorderCrop module."""
        module = BorderCrop(borders=4)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 24, 24)

    def test_callable_borders(self):
        """borders can be a callable for dynamic cropping."""
        module = BorderCrop(borders=lambda: 2)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 28, 28)

    def test_partial_kwargs(self):
        """Forward can provide missing kwargs (borders required)."""
        module = BorderCrop()
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # borders not set in module, provided at forward time
        result = module(x, borders=2)
        assert result.shape == (2, 3, 28, 28)

    def test_zero_crop(self):
        """Zero crop should return the same tensor."""
        module = BorderCrop(borders=0)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == x.shape
        assert torch.allclose(result, x)
