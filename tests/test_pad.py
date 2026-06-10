"""Tests for planar_transforms pad transform."""

import pytest
import torch

from planar_transforms import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    OrientedBoxSet,
    Pad,
    PointSet,
    VectorSet,
)
from planar_transforms.functional.pad import pad


class TestPadFunctional:
    """Tests for the functional pad API."""

    def test_continuous_field_uniform_pad(self):
        """Pad a continuous field uniformly on all sides."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = pad(x, pad=4)
        assert result.shape == (2, 3, 40, 40)
        assert isinstance(result, ContinuousField)

    def test_continuous_field_asymmetric_pad(self):
        """Pad a continuous field with different amounts per side."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # (left, right, top, bottom)
        result = pad(x, pad=[1, 2, 3, 4])
        assert result.shape == (2, 3, 39, 35)  # H: 32+3+4=39, W: 32+1+2=35

    def test_discrete_field_pad(self):
        """Pad a discrete field."""
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        result = pad(x, pad=2)
        assert result.shape == (2, 1, 36, 36)
        assert isinstance(result, DiscreteField)

    def test_negative_pad_crops(self):
        """Negative padding should crop the tensor."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = pad(x, pad=-4)
        assert result.shape == (2, 3, 24, 24)

    def test_mixed_pad_and_crop(self):
        """Mix of positive and negative padding."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # left=+2, right=-4, top=+6, bottom=-2
        result = pad(x, pad=[2, -4, 6, -2])
        # W: 32+2-4=30, H: 32+6-2=36
        assert result.shape == (2, 3, 36, 30)

    def test_pad_with_value(self):
        """Padding with constant value."""
        x = ContinuousField(torch.zeros(1, 1, 4, 4))
        result = pad(x, pad=1, mode="constant", value=1.0)
        assert result.shape == (1, 1, 6, 6)
        # Check corners are filled with the value
        assert result[0, 0, 0, 0].item() == 1.0
        assert result[0, 0, 0, -1].item() == 1.0
        assert result[0, 0, -1, 0].item() == 1.0
        assert result[0, 0, -1, -1].item() == 1.0

    def test_pad_with_reflect_mode(self):
        """Padding with reflect mode."""
        x = ContinuousField(torch.arange(16).reshape(1, 1, 4, 4).float())
        result = pad(x, pad=1, mode="reflect")
        assert result.shape == (1, 1, 6, 6)

    def test_vectorset_translation(self):
        """VectorSet should translate by half the pad difference."""
        x = VectorSet(torch.tensor([[10.0, 20.0]]))
        # left=4, right=0, top=0, bottom=2 -> translation = (4-0)/2, (2-0)/2 = (2, 1)
        result = pad(x, pad=[4, 0, 0, 2])
        expected = torch.tensor([[12.0, 21.0]])
        assert torch.allclose(result, expected)

    def test_pointset_translation(self):
        """PointSet should translate by half the pad difference."""
        x = PointSet(torch.tensor([[0.0, 0.0]]))
        # Symmetric padding: no translation expected
        result = pad(x, pad=4)
        assert torch.allclose(result, x)

    def test_boxset_translation(self):
        """BoxSet centroids should translate while radii stay unchanged."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        x = BoxSet(radii=radii, centroids=centroids)

        # left=10, right=0, top=0, bottom=6 -> translation = (5, 3)
        result = pad(x, pad=[10, 0, 0, 6])

        assert torch.allclose(result.radii, radii)  # unchanged
        assert torch.allclose(result.centroids, torch.tensor([[25.0, 33.0]]))

    def test_oriented_boxset_translation(self):
        """OrientedBoxSet centroids should translate while radii and rotors stay unchanged."""
        radii = torch.tensor([[5.0, 10.0]])
        centroids = torch.tensor([[20.0, 30.0]])
        rotors = torch.tensor([[1.0, 0.0]])
        x = OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)

        result = pad(x, pad=[10, 0, 0, 6])

        assert torch.allclose(result.radii, radii)
        assert torch.allclose(result.centroids, torch.tensor([[25.0, 33.0]]))
        assert torch.allclose(result.rotors, rotors)

    def test_tensor_pad_values(self):
        """Padding can be specified as a tensor."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        pad_tensor = torch.tensor([1, 2, 3, 4])
        result = pad(x, pad=pad_tensor)
        assert result.shape == (2, 3, 39, 35)

    def test_float_pad_raises(self):
        """Floating point padding values should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(AssertionError, match="integer"):
            pad(x, pad=torch.tensor([1.5, 2.0, 3.0, 4.0]))

    def test_excessive_crop_raises(self):
        """Cropping more than image size should raise."""
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        with pytest.raises(ValueError, match="negative dimensions"):
            pad(x, pad=-20)  # Would result in negative dimensions

    def test_batch_dimensions(self):
        """Padding should preserve batch dimensions."""
        x = ContinuousField(torch.randn(4, 2, 3, 32, 32))
        result = pad(x, pad=2)
        assert result.shape == (4, 2, 3, 36, 36)


class TestPadModule:
    """Tests for the Pad nn.Module wrapper."""

    def test_module_forward(self):
        """Basic forward pass through Pad module."""
        module = Pad(pad=4)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 40, 40)

    def test_callable_pad(self):
        """pad can be a callable for dynamic padding."""
        module = Pad(pad=lambda: 2)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        result = module(x)
        assert result.shape == (2, 3, 36, 36)

    def test_partial_kwargs(self):
        """Forward can provide missing kwargs."""
        module = Pad(mode="constant", value=0.0)
        x = ContinuousField(torch.randn(2, 3, 32, 32))
        # pad not set in module, provided at forward time
        result = module(x, pad=2)
        assert result.shape == (2, 3, 36, 36)

    def test_mode_and_value(self):
        """Module should respect mode and value options."""
        module = Pad(pad=2, mode="constant", value=0.5)
        x = ContinuousField(torch.zeros(1, 1, 4, 4))
        result = module(x)
        assert result.shape == (1, 1, 8, 8)
        assert result[0, 0, 0, 0].item() == 0.5
