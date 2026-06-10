"""Regression test: geometry rotation must track the image rotation.

The image path (ContinuousField) rotates via torchvision; geometry rotates in the r2
frame. With the pixel<->r2 boundary carrying the handedness, the full round trip

    to_pixel(rotate(from_pixel(p), theta)) == image_rotate(p, theta)

must hold. This is the design-agnostic spec for the coordinate frames: if it fails, a
convention (the r2 rotation, or the frame conversion) is wrong.
"""

import pytest
import torch

from planar_transforms import pixel, r2
from planar_transforms.functional.rotate import rotate
from planar_transforms.types import ContinuousField


def _centroid_rowcol(img2d):
    """Intensity-weighted (row, col) centroid of an HxW tensor."""
    img2d = img2d.as_subclass(torch.Tensor)
    H, W = img2d.shape
    rr, cc = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    total = img2d.sum()
    return float((img2d * rr).sum() / total), float((img2d * cc).sum() / total)


@pytest.mark.parametrize("angle", [30.0, 90.0, 150.0, -60.0, 215.0])
def test_geometry_rotation_tracks_image_rotation(angle):
    H = W = 41

    # asymmetric marker, up-and-right of centre (no symmetry to hide a flip)
    img = torch.zeros(1, 1, H, W)
    img[0, 0, 6:11, 27:32] = 1.0

    fr, fc = _centroid_rowcol(img[0, 0])
    # feature as a pixel-frame point (row, col), converted into r2
    p_r2 = r2.from_pixel(pixel.PointSet(torch.tensor([[fr, fc]])), size=(H, W))

    # ground truth: where torchvision moves the marker
    rimg = rotate(ContinuousField(img), angle=float(angle), interpolation="bilinear")
    gr, gc = _centroid_rowcol(rimg[0, 0])

    # geometry path: rotate in r2, convert back to pixel
    rotated = rotate(p_r2, angle=float(angle))
    back = r2.to_pixel(rotated, size=(H, W))
    pred_row, pred_col = back[0].tolist()

    assert abs(pred_row - gr) < 1.0 and abs(pred_col - gc) < 1.0, (
        f"angle={angle}: geometry ({pred_row:.2f},{pred_col:.2f}) does not track "
        f"image ({gr:.2f},{gc:.2f})"
    )
