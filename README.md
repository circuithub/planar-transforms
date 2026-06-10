# planar-transforms

PyTorch transforms for planar (2D) data. Supports images, segmentation masks, points, and bounding boxes with consistent coordinate handling.

> ⚠️ **Experimental.** This library is under active development, its API may change without notice, and we are unlikely to take outside contributions for the time being. Use at your own risk.

## Installation

```bash
pip install planar-transforms
```

## Usage

### Basic Transforms

```python
import torch
from planar_transforms import Resize, Rotate, Pad, BorderCrop, ContinuousField

# Create an image tensor
image = ContinuousField(torch.randn(1, 3, 256, 256))

# Resize
resize = Resize(original_size=(256, 256), new_size=(128, 128))
resized = resize(image)

# Rotate (geometry tracks the image; image-space coords, y down)
rotate = Rotate(angle=45.0)
rotated = rotate(image)

# Pad (positive values pad, negative values crop)
pad = Pad(pad=16)
padded = pad(image)

# Border crop
crop = BorderCrop(borders=32)
cropped = crop(image)
```

### Functional API

```python
from planar_transforms import functional as F, ContinuousField

image = ContinuousField(torch.randn(1, 3, 256, 256))

resized = F.resize(image, original_size=(256, 256), new_size=(128, 128))
rotated = F.rotate(image, angle=45.0)
padded = F.pad(image, pad=16)
cropped = F.border_crop(image, borders=32)
```

### Tensor Types

The library provides specialized tensor types that enable type-aware transform behavior:

```python
from planar_transforms import (
    ContinuousField,  # Interpolatable data (images)
    DiscreteField,    # Non-interpolatable data (segmentation masks)
    ImageTensor,      # Images clamped to [0,1] or [0,255]
    PointSet,         # Sets of 2D points
    BoxSet,           # Axis-aligned bounding boxes
    OrientedBoxSet,   # Rotated bounding boxes
)

# ContinuousField uses bilinear interpolation
image = ContinuousField(torch.randn(1, 3, 256, 256))
resized_image = F.resize(image, original_size=(256, 256), new_size=(128, 128))

# DiscreteField uses nearest-neighbor (no interpolation)
mask = DiscreteField(torch.randint(0, 10, (1, 1, 256, 256)).float())
resized_mask = F.resize(mask, original_size=(256, 256), new_size=(128, 128))

# Points scale with the image
points = PointSet(torch.tensor([[100.0, 100.0], [200.0, 150.0]]))
resized_points = F.resize(points, original_size=(256, 256), new_size=(128, 128))
# -> [[50.0, 50.0], [100.0, 75.0]]
```

### Compositing

```python
from planar_transforms import compositing, ContinuousField

target = ContinuousField(torch.zeros(1, 3, 256, 256))
source = ContinuousField(torch.ones(1, 3, 64, 64))

# Paste source onto target at offset
result = compositing.paste(
    target, source,
    offset=(100, 100),
    blend=lambda t, s: s  # Replace blend
)

# Alpha blending
from planar_transforms.compositing import alpha_blend_, Direction

target_rgba = ContinuousField(torch.rand(1, 4, 64, 64))
source_rgba = ContinuousField(torch.rand(1, 4, 64, 64))
alpha_blend_(target_rgba, source_rgba, Direction.back_to_front)
```

### Experimental: RandomState

```python
from planar_transforms.experimental import RandomState

state = RandomState()

# Define random parameters
angle = state.float("angle", low=-30.0, high=30.0)
size = state.size("crop_size", low=(100, 100), high=(200, 200))

# Use in transforms (values are cached until regenerate())
rotate = Rotate(angle=angle)
result = rotate(image)

# Regenerate all random values for next sample
state.regenerate()
```

## Development

```bash
nix develop
pytest tests/
black src/ tests/
ruff check src/ tests/
mypy src/
```

## License

MIT
