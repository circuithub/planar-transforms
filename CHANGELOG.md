# Changelog

## Unreleased

### Added

- **Batched affine transform** (`functional.affine`, `Affine` module). One
  `grid_sample` call applies per-sample angle / translate / scale / shear to a whole
  batch on the GPU, replacing per-item resampling loops. Continuous fields interpolate
  (bilinear), discrete fields stay nearest and reject bilinear. Pure-rotation output
  matches `rotate`; angles are CCW (r2 convention). Matches the per-item torchvision
  affine to sub-pixel accuracy.
- **Field-typed `flip`** (`functional.flip`, `Flip` module). Reflects a field across a
  spatial axis, dispatched on field type: continuous and discrete fields reverse the
  spatial index (exact, no interpolation), sets reflect the flipped r2 axis, and oriented
  boxes also conjugate the rotor. A per-sample boolean mask selects which batch elements
  flip. Render-verified against an image flip in `tests/test_image_geometry_alignment.py`.

### Fixed

- **Rotation direction for geometry.** The 2D rotation matrix is transposed so that
  points, boxes, and oriented boxes rotate to match the image rotation in image-space
  coordinates (origin top-left, y down). Across angles the geometry tracks the rotated
  image to sub-pixel accuracy; the non-transposed form diverged by many pixels. Verified
  by `tests/test_image_geometry_alignment.py`.
- **BoxSet resize scaling.** A `BoxSet` is `(radii_x, radii_y, centroid_x, centroid_y)`
  (shape `…,4`) while the scale factor is shape `(2,)`; scaling now tiles it via
  `scale_factor.repeat(2)`. A bare multiply raised a `(4 vs 2)` broadcast error. Covered
  by a non-square `test_boxset_scaling`.
