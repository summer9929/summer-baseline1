"""Pixel-exact RGB results shared by validation and prediction."""

from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import restore_mask


def make_visuals(cropped_rgb, pred, decode_target, target=None):
    image = np.asarray(cropped_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 HWC RGB')
    pred = restore_mask(pred, image.shape[:2])
    color = decode_target(pred).astype(np.float32)
    overlay = np.clip(np.rint(0.6 * image.astype(np.float32) + 0.4 * color), 0, 255).astype(np.uint8)
    visuals = {'image': image, 'pred_id': pred, 'overlay': overlay}
    if target is not None:
        target = np.asarray(target)
        if target.shape != image.shape[:2]:
            raise ValueError('Cropped ground truth does not align with RGB')
        visuals['target_color'] = decode_target(target)
        visuals['target_id'] = target.astype(np.uint8)
    return visuals


def save_index_mask(path, class_ids, gray_values):
    """Keep 0/1/2 pixel indices and attach black/white/gray PNG display colors."""
    ids = np.asarray(class_ids)
    shades = np.asarray(gray_values)
    if (ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer)
            or shades.ndim != 1 or not 1 <= len(shades) <= 256
            or not np.issubdtype(shades.dtype, np.integer)
            or np.any(shades < 0) or np.any(shades > 255)
            or np.any(ids < 0) or np.any(ids >= len(shades))):
        raise ValueError('Expected integer class IDs and a matching uint8 grayscale palette')
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:len(shades)] = shades[:, None]
    image = Image.frombytes('P', (ids.shape[1], ids.shape[0]),
                            np.ascontiguousarray(ids, dtype=np.uint8).tobytes())
    image.putpalette(palette.ravel().tolist())
    image.save(path)


def save_sample_results(output_dir, sample_id, visuals, mask_gray_values):
    """Five types with GT, three without GT; no separate prediction-color PNG."""
    prefix = Path(output_dir) / sample_id
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, array in visuals.items():
        path = prefix.parent / ('%s_%s.png' % (prefix.name, name))
        if name in ('pred_id', 'target_id'):
            save_index_mask(path, array, mask_gray_values)
        else:
            Image.fromarray(array).save(path)
