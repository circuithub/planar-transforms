# Changelog

## Unreleased

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
