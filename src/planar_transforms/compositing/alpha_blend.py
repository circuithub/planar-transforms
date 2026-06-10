from enum import Enum

from planar_transforms.types import ImageTensor


class Direction(Enum):
    back_to_front = "back_to_front"
    front_to_back = "front_to_back"


def alpha_blend_(target: ImageTensor, source: ImageTensor, direction: Direction | str):
    """
    Perform inplace alpha compositing.
    Alpha compositing can be done back-to-front such that the source image is placed on top of the target image,
    or front-to-back such that the source image is placed behind the target image.
    """
    # Narrow down the alpha channels
    target_alpha = target[..., -1:, :, :]
    source_alpha = source[..., -1:, :, :]

    # Narrow down the color channels
    target_value = target[..., :-1, :, :]
    source_value = source[..., :-1, :, :]

    if direction in (Direction.front_to_back, Direction.front_to_back.value):
        target_reverse_alpha = 1.0 - target_alpha
        target_value[...] = (
            # Target in front
            target_alpha * target_value
            # Source behind
            + (source_alpha * target_reverse_alpha) * source_value
        )
        target_alpha[...] = (
            # Target in front
            target_alpha
            # Source behind
            + source_alpha * target_reverse_alpha
        )
    elif direction in (Direction.back_to_front, Direction.back_to_front.value):
        source_reverse_alpha = 1.0 - source_alpha
        target_value[...] = (
            # Source in front
            source_alpha * source_value
            # Target behind
            + (target_alpha * source_reverse_alpha) * target_value
        )
        target_alpha[...] = (
            # Source in front
            source_alpha
            # Target behind
            + target_alpha * source_reverse_alpha
        )
    else:
        raise ValueError(
            f"Expected compositing direction {tuple(dir.value for dir in Direction)}, got {direction}"
        )
    return target
