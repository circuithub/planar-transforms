"""Tests for planar_transforms flip transform."""

import pytest
import torch

from planar_transforms import (
    ContinuousField,
    DiscreteField,
    Flip,
    r2,
)
from planar_transforms.functional.flip import flip


class TestFlipFunctional:
    def test_continuous_horizontal_reverses_columns(self):
        x = ContinuousField(torch.arange(2 * 3 * 4 * 5).float().reshape(2, 3, 4, 5))
        result = flip(x, axis="horizontal")
        assert isinstance(result, ContinuousField)
        assert torch.equal(result, x.flip(dims=(-1,)))

    def test_continuous_vertical_reverses_rows(self):
        x = ContinuousField(torch.randn(2, 3, 4, 5))
        result = flip(x, axis="vertical")
        assert torch.equal(result, x.flip(dims=(-2,)))

    def test_discrete_preserves_labels(self):
        x = DiscreteField(torch.randint(0, 7, (2, 1, 8, 8)))
        result = flip(x, axis="horizontal")
        assert isinstance(result, DiscreteField)
        assert result.dtype == x.dtype
        assert set(result.unique().tolist()) == set(x.unique().tolist())

    def test_mask_selects_samples(self):
        x = ContinuousField(torch.randn(3, 1, 4, 4))
        mask = torch.tensor([True, False, True])
        result = flip(x, axis="horizontal", mask=mask)
        assert torch.equal(result[0], x[0].flip(dims=(-1,)))
        assert torch.equal(result[1], x[1])  # untouched
        assert torch.equal(result[2], x[2].flip(dims=(-1,)))

    def test_pointset_horizontal_negates_x(self):
        p = r2.PointSet(torch.tensor([[3.0, 5.0]]))
        result = flip(p, axis="horizontal")
        assert torch.allclose(result, torch.tensor([[-3.0, 5.0]]))
        assert isinstance(result, r2.PointSet)

    def test_pointset_vertical_negates_y(self):
        p = r2.PointSet(torch.tensor([[3.0, 5.0]]))
        result = flip(p, axis="vertical")
        assert torch.allclose(result, torch.tensor([[3.0, -5.0]]))

    def test_boxset_reflects_centroid_keeps_radii(self):
        box = r2.BoxSet(radii=torch.tensor([[5.0, 10.0]]), centroids=torch.tensor([[8.0, 2.0]]))
        result = flip(box, axis="horizontal")
        assert torch.allclose(result.radii, box.radii)
        assert torch.allclose(result.centroids, torch.tensor([[-8.0, 2.0]]))

    def test_oriented_box_conjugates_rotor(self):
        theta = torch.deg2rad(torch.tensor(40.0))
        rotor = torch.tensor([[torch.cos(theta / 2), torch.sin(theta / 2)]])
        box = r2.OrientedBoxSet(
            radii=torch.tensor([[5.0, 10.0]]), centroids=torch.tensor([[8.0, 2.0]]), rotors=rotor
        )
        result = flip(box, axis="horizontal")
        assert torch.allclose(result.rotors, rotor * torch.tensor([1.0, -1.0]))
        assert torch.allclose(result.centroids, torch.tensor([[-8.0, 2.0]]))

    def test_per_sample_mask_on_boxes(self):
        box = r2.BoxSet(
            radii=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            centroids=torch.tensor([[4.0, 0.0], [6.0, 0.0]]),
        )
        result = flip(box, axis="horizontal", mask=torch.tensor([True, False]))
        assert torch.allclose(result.centroids, torch.tensor([[-4.0, 0.0], [6.0, 0.0]]))

    def test_rejects_pixel_frame(self):
        from planar_transforms import pixel

        with pytest.raises(NotImplementedError):
            flip(pixel.PointSet(torch.tensor([[1.0, 0.0]])))

    def test_rejects_bad_axis(self):
        with pytest.raises(ValueError, match="axis"):
            flip(ContinuousField(torch.randn(1, 1, 4, 4)), axis="diagonal")  # type: ignore[arg-type]


class TestFlipModule:
    def test_module_forward(self):
        module = Flip(axis="horizontal")
        x = ContinuousField(torch.randn(2, 3, 8, 8))
        assert torch.equal(module(x), x.flip(dims=(-1,)))

    def test_callable_mask(self):
        module = Flip(axis="horizontal", mask=lambda: torch.tensor([True, False]))
        x = ContinuousField(torch.randn(2, 1, 4, 4))
        result = module(x)
        assert torch.equal(result[1], x[1])
