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
from planar_transforms.functional.flip import flip
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


def _render_oriented_box(obox_r2, H, W):
    """Rasterise a single r2 OrientedBoxSet into an HxW mask via its pixel-frame corners.

    This is the design-agnostic check for flip's box/rotor handling: a rendered flip must
    match flipping the rendered box, which catches any centroid- or rotor-sign error.
    """
    rot = obox_r2.rotors[0]
    theta = 2.0 * torch.atan2(rot[1], rot[0])
    c, s = torch.cos(theta), torch.sin(theta)
    rotation = torch.tensor([[c, -s], [s, c]])
    local = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    corners_r2 = (rotation @ (local * obox_r2.radii[0]).T).T + obox_r2.centroids[0]
    corners_px = r2.to_pixel(r2.PointSet(corners_r2), size=(H, W))

    rr, cc = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    crosses = []
    for k in range(4):
        a, b = corners_px[k], corners_px[(k + 1) % 4]
        edge = b - a
        crosses.append(edge[0] * (cc - a[1]) - edge[1] * (rr - a[0]))
    stacked = torch.stack(crosses)
    inside = (stacked >= 0).all(0) | (stacked <= 0).all(0)
    return inside.float()


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_flip_oriented_box_tracks_image_flip(axis):
    """Rendering a flipped oriented box must match flipping the rendered box."""
    H = W = 81
    theta = torch.deg2rad(torch.tensor(35.0))
    box = r2.OrientedBoxSet(
        radii=torch.tensor([[18.0, 6.0]]),
        centroids=torch.tensor([[20.0, 12.0]]),  # off-centre: no symmetry hides a sign flip
        rotors=torch.tensor([[torch.cos(theta / 2), torch.sin(theta / 2)]]),
    )

    rendered = _render_oriented_box(box, H, W)
    image_dim = -1 if axis == "horizontal" else -2
    expected = rendered.flip(dims=(image_dim,))

    flipped_box = flip(box, axis=axis)
    geometry = _render_oriented_box(flipped_box, H, W)

    iou = float((geometry * expected).sum() / ((geometry + expected) > 0).float().sum())
    assert iou > 0.97, f"axis={axis}: flipped box IoU {iou:.3f} does not track the image flip"
