"""Tests for planar_transforms affine and joint_affine transforms."""

import pytest
import torch
import torchvision

from planar_transforms import Affine, ContinuousField, Degrees, DiscreteField
from planar_transforms.functional.affine import affine
from planar_transforms.functional.joint import joint_affine
from planar_transforms.functional.rotate import rotate


def _tv_affine(img, angle, shear, interp):
    """Reference: per-item torchvision affine. CCW angle/shear -> negate for torchvision."""
    B = img.shape[0]
    return torch.stack(
        [
            torchvision.transforms.functional.affine(
                img[i],
                angle=-float(angle[i] if torch.is_tensor(angle) else angle),
                translate=[0, 0],
                scale=1.0,
                shear=[
                    -float(shear[i][0] if torch.is_tensor(shear) else shear[0]),
                    -float(shear[i][1] if torch.is_tensor(shear) else shear[1]),
                ],
                interpolation=interp,
            )
            for i in range(B)
        ]
    )


class TestAffineFunctional:
    def test_identity(self):
        x = ContinuousField(torch.rand(2, 3, 32, 32))
        result = affine(x)
        assert torch.allclose(result, x, atol=1e-4)
        assert isinstance(result, ContinuousField)

    def test_matches_torchvision_affine_bilinear(self):
        """One batched grid_sample call matches the per-item torchvision affine loop."""
        torch.manual_seed(0)
        x = ContinuousField(torch.rand(4, 3, 64, 64))
        angle = torch.tensor([10.0, -7.0, 23.0, 0.0])
        shear = torch.tensor([[5.0, -3.0], [0.0, 8.0], [-10.0, 4.0], [0.0, 0.0]])
        got = affine(x, angle=angle, shear=shear)
        ref = _tv_affine(x, angle, shear, torchvision.transforms.InterpolationMode.BILINEAR)
        inner = (got - ref).abs()[..., 4:-4, 4:-4]
        assert inner.mean() < 1e-3 and inner.max() < 1e-2

    def test_pure_rotation_matches_rotate(self):
        """Affine with only an angle matches the existing rotate transform (CCW)."""
        torch.manual_seed(1)
        x = ContinuousField(torch.rand(3, 3, 48, 48))
        angle = torch.tensor([30.0, -15.0, 90.0])
        got = affine(x, angle=angle)
        ref = rotate(x, angle=angle)
        inner = (got - ref).abs()[..., 3:-3, 3:-3]
        assert inner.mean() < 1e-3 and inner.max() < 5e-2

    def test_discrete_field_uses_nearest(self):
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        result = affine(x, angle=Degrees(90.0))
        assert isinstance(result, DiscreteField)
        assert result.dtype == x.dtype

    def test_discrete_field_rejects_bilinear(self):
        x = DiscreteField(torch.randint(0, 10, (2, 1, 32, 32)))
        with pytest.raises(AssertionError, match="nearest"):
            affine(x, angle=Degrees(45.0), interpolation="bilinear")

    def test_discrete_field_preserves_label_values(self):
        """Nearest resampling must not invent label ids."""
        x = DiscreteField(torch.randint(0, 5, (2, 1, 40, 40)))
        result = affine(x, angle=Degrees(33.0))
        present = set(x.unique().tolist())
        assert set(result.unique().tolist()).issubset(present)

    def test_per_sample_angles(self):
        """Different angles per batch item in a single call."""
        x = ContinuousField(torch.rand(3, 1, 32, 32))
        angle = torch.tensor([0.0, 45.0, 90.0])
        result = affine(x, angle=angle)
        # Sample 0 is identity; samples 1 and 2 are not.
        assert torch.allclose(result[0], x[0], atol=1e-4)
        assert not torch.allclose(result[2], x[2], atol=1e-2)

    def test_unbatched_chw(self):
        x = ContinuousField(torch.rand(3, 32, 32))
        result = affine(x, angle=Degrees(20.0))
        assert result.shape == (3, 32, 32)

    def test_rejects_geometry(self):
        from planar_transforms import r2

        with pytest.raises(NotImplementedError):
            affine(r2.PointSet(torch.tensor([[1.0, 0.0]])), angle=Degrees(90.0))


class TestJointAffine:
    def test_image_and_mask_move_together(self):
        """Image and label map under one set of params; mask stays discrete."""
        torch.manual_seed(2)
        img = ContinuousField(torch.rand(2, 3, 48, 48))
        mask = DiscreteField(torch.randint(0, 4, (2, 1, 48, 48)))
        angle = torch.tensor([25.0, -40.0])
        out_img, out_mask = joint_affine(img, mask, angle=angle, shear=[6.0, -2.0])
        assert isinstance(out_img, ContinuousField)
        assert isinstance(out_mask, DiscreteField)
        # Mask labels are preserved (nearest), image is interpolated.
        assert set(out_mask.unique().tolist()).issubset(set(mask.unique().tolist()))
        assert out_img.shape == img.shape and out_mask.shape == mask.shape

    def test_consistency_with_separate_calls(self):
        img = ContinuousField(torch.rand(2, 3, 32, 32))
        mask = DiscreteField(torch.randint(0, 3, (2, 1, 32, 32)))
        angle = torch.tensor([15.0, 30.0])
        ji, jm = joint_affine(img, mask, angle=angle)
        si = affine(img, angle=angle)
        sm = affine(mask, angle=angle)
        assert torch.allclose(ji, si) and torch.equal(jm, sm)

    def test_mismatched_shapes_rejected(self):
        img = ContinuousField(torch.rand(2, 3, 32, 32))
        mask = DiscreteField(torch.randint(0, 3, (2, 1, 16, 16)))
        with pytest.raises(AssertionError, match="height and width"):
            joint_affine(img, mask, angle=Degrees(10.0))


class TestAffineModule:
    def test_module_forward(self):
        module = Affine(angle=45.0, shear=[5.0, 5.0])
        x = ContinuousField(torch.rand(2, 3, 32, 32))
        assert module(x).shape == (2, 3, 32, 32)

    def test_callable_params(self):
        module = Affine(angle=lambda: torch.tensor([10.0, 20.0]))
        x = ContinuousField(torch.rand(2, 3, 32, 32))
        assert module(x).shape == (2, 3, 32, 32)
