"""Tests for planar_transforms.types module."""

import torch

from planar_transforms import (
    BoxSet,
    ContinuousField,
    DiscreteField,
    ImageTensor,
    OrientedBoxSet,
    PointSet,
    TensorLike,
    VectorSet,
)


class TestTensorLike:
    def test_subclass_of_tensor(self):
        assert issubclass(TensorLike, torch.Tensor)

    def test_torch_where_preserves_type(self):
        """TensorLike should work with torch.where without errors."""

        class MyTensor(TensorLike):
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

        x = MyTensor(torch.tensor([1.0, 2.0, 3.0]))
        y = MyTensor(torch.tensor([4.0, 5.0, 6.0]))
        cond = torch.tensor([True, False, True])

        result = torch.where(cond, x, y)
        assert isinstance(result, torch.Tensor)


class TestContinuousField:
    def test_empty_construction(self):
        field = ContinuousField()
        assert field.numel() == 0

    def test_from_tensor(self):
        data = torch.randn(2, 3, 32, 32)
        field = ContinuousField(data)
        assert field.shape == (2, 3, 32, 32)
        assert isinstance(field, ContinuousField)
        assert isinstance(field, TensorLike)

    def test_batch_sizes(self):
        data = torch.randn(4, 2, 3, 32, 32)
        field = ContinuousField(data)
        assert field.batch_sizes == torch.Size([4, 2])


class TestDiscreteField:
    def test_empty_construction(self):
        field = DiscreteField()
        assert field.numel() == 0

    def test_from_tensor(self):
        data = torch.randint(0, 10, (2, 1, 32, 32))
        field = DiscreteField(data)
        assert field.shape == (2, 1, 32, 32)
        assert isinstance(field, DiscreteField)
        assert isinstance(field, TensorLike)

    def test_batch_sizes(self):
        data = torch.randint(0, 10, (4, 2, 1, 32, 32))
        field = DiscreteField(data)
        assert field.batch_sizes == torch.Size([4, 2])


class TestImageTensor:
    def test_empty_construction(self):
        img = ImageTensor()
        assert img.numel() == 0

    def test_float_image(self):
        data = torch.rand(2, 3, 32, 32)
        img = ImageTensor(data)
        assert img.shape == (2, 3, 32, 32)
        assert isinstance(img, ImageTensor)
        assert isinstance(img, ContinuousField)

    def test_uint8_image(self):
        data = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        img = ImageTensor(data)
        assert img.dtype == torch.uint8

    def test_invalid_dtype_raises(self):
        data = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.int32)
        try:
            ImageTensor(data)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass


class TestVectorSet:
    def test_empty_construction(self):
        vs = VectorSet()
        assert vs.numel() == 0

    def test_from_tensor(self):
        data = torch.randn(10, 64)
        vs = VectorSet(data)
        assert vs.shape == (10, 64)

    def test_batch_sizes(self):
        data = torch.randn(4, 10, 64)
        vs = VectorSet(data)
        assert vs.batch_sizes == torch.Size([4, 10])


class TestPointSet:
    def test_empty_construction(self):
        ps = PointSet()
        assert ps.numel() == 0

    def test_inherits_vectorset(self):
        assert issubclass(PointSet, VectorSet)


class TestBoxSet:
    def test_empty_construction(self):
        bs = BoxSet()
        assert bs.numel() == 0

    def test_from_data(self):
        data = torch.randn(10, 4)
        bs = BoxSet(data)
        assert bs.shape == (10, 4)

    def test_from_radii_centroids(self):
        radii = torch.randn(10, 2)
        centroids = torch.randn(10, 2)
        bs = BoxSet(radii=radii, centroids=centroids)
        assert bs.shape == (10, 4)
        assert torch.allclose(bs.radii, radii)
        assert torch.allclose(bs.centroids, centroids)

    def test_batch_sizes(self):
        data = torch.randn(4, 10, 4)
        bs = BoxSet(data)
        assert bs.batch_sizes == torch.Size([4, 10])


class TestOrientedBoxSet:
    def test_empty_construction(self):
        obs = OrientedBoxSet()
        assert obs.numel() == 0

    def test_from_data(self):
        data = torch.randn(10, 6)
        obs = OrientedBoxSet(data)
        assert obs.shape == (10, 6)

    def test_from_components(self):
        radii = torch.randn(10, 2)
        centroids = torch.randn(10, 2)
        rotors = torch.randn(10, 2)
        obs = OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)
        assert obs.shape == (10, 6)
        assert torch.allclose(obs.radii, radii)
        assert torch.allclose(obs.centroids, centroids)
        assert torch.allclose(obs.rotors, rotors)

    def test_unoriented(self):
        data = torch.randn(10, 6)
        obs = OrientedBoxSet(data)
        unoriented = obs.unoriented
        assert isinstance(unoriented, BoxSet)
        assert unoriented.shape == (10, 4)

    def test_inherits_boxset(self):
        assert issubclass(OrientedBoxSet, BoxSet)
