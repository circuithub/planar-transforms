"""Regression test: geometry rotation must track the image rotation.

The image path (ContinuousField) rotates via torchvision; the geometry path
(PointSet/BoxSet centroids) rotates via a matrix. These MUST agree, otherwise
rotation augmentation silently misaligns labels from the pixels.

This is the test the "remove the transpose / use math convention" change failed:
it makes the geometry rotate opposite to the image. We pin alignment directly by
rotating an asymmetric image feature and the point that marks it, then comparing.

Construction convention matches the consumer (objectdetector.py): a centroid is
``(box_center_xy) - image_size/2`` with ``xy = (col, row)`` and y increasing DOWN.
"""

import pytest
import torch

from planar_transforms.functional.rotate import rotate
from planar_transforms.types import ContinuousField, PointSet


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
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0  # centre in (col, row)

    # asymmetric marker, up-and-right of centre (no symmetry to hide a flip)
    img = torch.zeros(1, 1, H, W)
    img[0, 0, 6:11, 27:32] = 1.0

    fr, fc = _centroid_rowcol(img[0, 0])
    # centroid in consumer convention: (x=col-cx, y=row-cy), y DOWN, rotated about origin
    point = PointSet(torch.tensor([[[fc - cx, fr - cy]]]))

    # ground truth: where torchvision moves the marker
    rimg = rotate(ContinuousField(img), angle=float(angle), interpolation="bilinear")
    gr, gc = _centroid_rowcol(rimg[0, 0])

    rx, ry = rotate(point, angle=float(angle))[0, 0].tolist()
    pred_row, pred_col = ry + cy, rx + cx

    assert abs(pred_row - gr) < 1.0 and abs(pred_col - gc) < 1.0, (
        f"angle={angle}: geometry ({pred_row:.2f},{pred_col:.2f}) does not track "
        f"image ({gr:.2f},{gc:.2f}) -- rotation matrix convention is wrong"
    )
