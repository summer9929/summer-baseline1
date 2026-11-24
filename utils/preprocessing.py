"""Shared mechanics; dataset-specific settings are declared only in main.py."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image

from .ext_transforms import ExtCenterCrop


@dataclass(frozen=True)
class PreprocessConfig:
    source_size_hw: tuple
    crop_size_hw: tuple
    input_size_hw: tuple
    mean: tuple
    std: tuple
    input_channels: int = 4
    depth_scale: float = 256.0
    depth_mean: float = 2.161799
    depth_std: float = 2.206930

    def __post_init__(self):
        for size in (self.source_size_hw, self.crop_size_hw, self.input_size_hw):
            if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
                raise ValueError('Sizes must be two positive integers in HW order')
        if any(c > s for c, s in zip(self.crop_size_hw, self.source_size_hw)):
            raise ValueError('Crop size exceeds source size')
        if len(self.mean) != 3 or len(self.std) != 3 or any(v <= 0 for v in self.std):
            raise ValueError('Expected three means and positive standard deviations')
        if self.input_channels not in (3, 4):
            raise ValueError('Expected RGB (3) or RGB-D (4) channels')
        if (not np.isfinite([self.depth_scale, self.depth_mean, self.depth_std]).all()
                or self.depth_scale <= 0 or self.depth_std <= 0):
            raise ValueError('Depth scale/std must be finite and positive')

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}


def read_depth(path):
    """Read raw 16-bit PNG without an RGB/8-bit conversion."""
    with Image.open(path) as source:
        if source.format != 'PNG':
            raise ValueError('Depth must be a 16-bit single-channel PNG: %s' % path)
        raw = np.array(source)
    validate_depth(raw)
    return raw


def validate_depth(raw):
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError('Expected uint16 single-channel depth, got %s / %s' %
                         (raw.dtype, raw.shape))


def resize_depth(raw, config):
    """Exactly the RGB center crop, then nearest sampling; never scale distances."""
    validate_depth(raw)
    if raw.shape != config.source_size_hw:
        raise ValueError('Depth source HW %s differs from RGB source HW %s' %
                         (raw.shape, config.source_size_hw))
    box = ExtCenterCrop(config.crop_size_hw).get_box(config.source_size_hw[::-1])
    return np.array(Image.fromarray(raw).crop(box).resize(
        config.input_size_hw[::-1], Image.Resampling.NEAREST), copy=True)


def normalize_depth(raw, config):
    validate_depth(raw)
    depth_m = raw.astype(np.float32) / config.depth_scale
    valid = (raw > 0) & np.isfinite(depth_m)
    normalized = np.zeros(raw.shape, dtype=np.float32)
    # All positive uint16 values are valid, including 65535. No numeric Clip.
    normalized[valid] = (depth_m[valid] - config.depth_mean) / config.depth_std
    return torch.from_numpy(normalized).unsqueeze(0)


def get_geometry(config):
    box = ExtCenterCrop(config.crop_size_hw).get_box(config.source_size_hw[::-1])
    return {
        'source_size_hw': np.array(config.source_size_hw, dtype=np.int64),
        'crop_box_ltrb': np.array(box, dtype=np.int64),
        'crop_size_hw': np.array(config.crop_size_hw, dtype=np.int64),
        'input_size_hw': np.array(config.input_size_hw, dtype=np.int64),
    }


class SegmentationPreprocessor:
    """Apply main's crop and input stages, optionally keeping full-size pixels."""

    def __init__(self, config, crop_transform, input_transform):
        self.config = config
        self.crop_transform = crop_transform
        self.input_transform = input_transform

    def __call__(self, image, target=None, return_metadata=False, *, depth=None):
        if image.size != self.config.source_size_hw[::-1]:
            raise ValueError('Expected source WH %s; got %s' %
                             (self.config.source_size_hw[::-1], image.size))
        if image.mode != 'RGB':
            raise ValueError('Expected RGB; got %s' % image.mode)
        if target is not None and image.size != target.size:
            raise ValueError('Image WH %s and label WH %s differ' % (image.size, target.size))
        depth_tensor = None
        if self.config.input_channels == 4:
            if depth is None:
                raise ValueError('RGB-D input requires a paired depth image')
            depth_tensor = normalize_depth(resize_depth(depth, self.config), self.config)
        image, target = self.crop_transform(image, target)
        metadata = None
        if return_metadata:
            metadata = {'cropped_rgb': np.array(image, dtype=np.uint8, copy=True),
                        'geometry': get_geometry(self.config)}
            if target is not None:
                metadata['cropped_target'] = np.array(target, dtype=np.uint8, copy=True)
        image, target = self.input_transform(image, target)
        if depth_tensor is not None:
            if image.shape[-2:] != depth_tensor.shape[-2:]:
                raise ValueError('RGB and depth transformed shapes differ')
            image = torch.cat((image, depth_tensor), dim=0)
        return (image, target, metadata) if return_metadata else (image, target)


def restore_mask(mask, crop_size_hw):
    """Nearest-neighbor resize of class IDs, never of an RGB palette image."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Expected a two-dimensional integer class mask')
    if np.any(mask < 0) or np.any(mask > 255):
        raise ValueError('Mask values must fit uint8')
    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(
        tuple(int(v) for v in crop_size_hw[::-1]), Image.Resampling.NEAREST), copy=True)
