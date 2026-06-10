"""Tests for the pixel <-> r2 frame conversions."""

import torch

from planar_transforms import pixel, r2


def _sorted_corners(c):
    """Sort a (4, 2) corner set into a canonical order for set comparison."""
    # round to avoid float jitter in the sort key
    keyed = [tuple(round(v, 4) for v in pt) for pt in c.tolist()]
    return sorted(keyed)


class TestPointVectorConversion:
    def test_point_known_values(self):
        # size (H=4, W=4): centre at (2, 2); top-left pixel (row=0, col=0) -> (-2, 2)
        p = pixel.PointSet(torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
        out = r2.from_pixel(p, size=(4, 4))
        assert torch.allclose(out, torch.tensor([[-2.0, 2.0], [0.0, 0.0]]))

    def test_vector_known_values(self):
        # moving +1 row (down) is -y in r2; +1 col (right) is +x
        v = pixel.VectorSet(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        out = r2.from_pixel(v, size=(8, 8))
        assert torch.allclose(out, torch.tensor([[0.0, -1.0], [1.0, 0.0]]))

    def test_point_roundtrip(self):
        p = pixel.PointSet(torch.randn(5, 2))
        back = r2.to_pixel(r2.from_pixel(p, (7, 11)), (7, 11))
        assert torch.allclose(back, p, atol=1e-5)

    def test_vector_roundtrip(self):
        v = pixel.VectorSet(torch.randn(5, 2))
        back = r2.to_pixel(r2.from_pixel(v, (7, 11)), (7, 11))
        assert torch.allclose(back, v, atol=1e-5)


class TestBoxConversion:
    def test_box_known_values(self):
        # radii (row=1, col=3) -> (x=3, y=1); centroid (row=1, col=1) @ (4,4) -> (-1, 1)
        b = pixel.BoxSet(radii=torch.tensor([[1.0, 3.0]]), centroids=torch.tensor([[1.0, 1.0]]))
        out = r2.from_pixel(b, size=(4, 4))
        assert torch.allclose(out.radii, torch.tensor([[3.0, 1.0]]))
        assert torch.allclose(out.centroids, torch.tensor([[-1.0, 1.0]]))

    def test_box_roundtrip(self):
        b = pixel.BoxSet(radii=torch.rand(4, 2) + 0.5, centroids=torch.randn(4, 2))
        back = r2.to_pixel(r2.from_pixel(b, (9, 13)), (9, 13))
        assert torch.allclose(back.radii, b.radii, atol=1e-5)
        assert torch.allclose(back.centroids, b.centroids, atol=1e-5)


def _full_angle(rotors):
    """Half-angle rotor (cos h, sin h) -> full-angle (cos, sin)."""
    c, s = rotors[..., 0], rotors[..., 1]
    return torch.stack([c * c - s * s, 2 * c * s], dim=-1)


def _corners(radii, centroids, rotors):
    """Corners of an oriented box: centroid + R(theta) . (+-r0, +-r1). Returns (...,4,2)."""
    cos, sin = _full_angle(rotors).unbind(dim=-1)  # (...,)
    R = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )  # (...,2,2)
    signs = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    offs = signs * radii[..., None, :]  # (...,4,2)
    rotated = torch.einsum("...ij,...kj->...ki", R, offs)  # (...,4,2)
    return centroids[..., None, :] + rotated


class TestOrientedBoxConversion:
    def test_oriented_roundtrip(self):
        radii = torch.rand(3, 2) + 0.5
        centroids = torch.randn(3, 2)
        theta = torch.tensor([0.3, 1.2, -2.0])
        rotors = torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], dim=-1)
        b = pixel.OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)
        back = r2.to_pixel(r2.from_pixel(b, (10, 14)), (10, 14))
        assert torch.allclose(back.radii, b.radii, atol=1e-5)
        assert torch.allclose(back.centroids, b.centroids, atol=1e-5)
        assert torch.allclose(back.rotors, b.rotors, atol=1e-5)

    def test_oriented_corners_consistent(self):
        # The definitive check: a pixel box's corners (in pixel space) must equal the
        # r2 box's corners mapped point-by-point to pixel space.
        size = (12, 16)
        radii = torch.tensor([[2.0, 1.0], [3.0, 0.5]])
        centroids = torch.tensor([[1.0, -2.0], [0.0, 4.0]])
        theta = torch.tensor([0.7, -1.1])
        rotors = torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], dim=-1)
        box_r2 = r2.OrientedBoxSet(radii=radii, centroids=centroids, rotors=rotors)

        # corners computed in r2, then each corner mapped (as a point) to pixel
        corners_r2 = _corners(box_r2.radii, box_r2.centroids, box_r2.rotors)  # (N,4,2)
        mapped = r2.to_pixel(r2.PointSet(corners_r2.reshape(-1, 2)), size).reshape(-1, 4, 2)

        # corners computed directly from the converted pixel box
        box_px = r2.to_pixel(box_r2, size)
        corners_px = _corners(
            box_px.radii.as_subclass(torch.Tensor),
            box_px.centroids.as_subclass(torch.Tensor),
            box_px.rotors.as_subclass(torch.Tensor),
        )

        for i in range(radii.shape[0]):
            assert _sorted_corners(mapped[i]) == _sorted_corners(corners_px[i])


class TestNormalized:
    def test_point_fill_scale(self):
        # size (4, 8): min/2 = 2, so divide by 2
        p = r2.PointSet(torch.tensor([[2.0, -4.0]]))
        out = r2.to_normalized(p, size=(4, 8))
        assert torch.allclose(out, torch.tensor([[1.0, -2.0]]))

    def test_box_scale_keeps_rotor(self):
        theta = torch.tensor([0.5])
        rotors = torch.stack([torch.cos(theta / 2), torch.sin(theta / 2)], dim=-1)
        b = r2.OrientedBoxSet(
            radii=torch.tensor([[2.0, 4.0]]), centroids=torch.tensor([[2.0, 2.0]]), rotors=rotors
        )
        out = r2.to_normalized(b, size=(4, 4))  # min/2 = 2
        assert torch.allclose(out.radii, torch.tensor([[1.0, 2.0]]))
        assert torch.allclose(out.centroids, torch.tensor([[1.0, 1.0]]))
        assert torch.allclose(out.rotors, rotors)
