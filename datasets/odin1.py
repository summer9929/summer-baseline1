"""odin1 loading and class semantics; preprocessing is supplied by main.py."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.preprocessing import read_depth


class Odin1(Dataset):
    splits = ('training', 'validation', 'testing')
    num_classes = 3
    class_names = ('不可通行', '可通行', '天空')
    raw_to_train_id = {0: 0, 255: 1, 128: 2}
    cmap = np.array([(220, 50, 50), (40, 200, 60), (60, 140, 240)], dtype=np.uint8)
    # Display class IDs using the dataset's original grayscale semantics.
    mask_gray_values = tuple(raw for raw, _ in sorted(raw_to_train_id.items(), key=lambda item: item[1]))

    def __init__(self, root, split='training', transform=None, return_metadata=False,
                 input_channels=None):
        if split not in self.splits:
            raise ValueError('Invalid odin1 split: %s' % split)
        if return_metadata and transform is None:
            raise ValueError('Returning cropped metadata requires a transform')
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata
        self.input_channels = (input_channels if input_channels is not None else
                               transform.config.input_channels if transform is not None else 4)
        if self.input_channels not in (3, 4):
            raise ValueError('Expected 3 or 4 input channels')
        if transform is not None and self.input_channels != transform.config.input_channels:
            raise ValueError('Dataset and transform input channels differ')
        images = self._index(self.root / split / 'image_data')
        targets = self._index(self.root / split / 'gt_image')
        if images.keys() != targets.keys():
            raise ValueError('%s IDs differ: missing labels=%s, extra labels=%s' %
                             (split, sorted(images.keys() - targets.keys()),
                              sorted(targets.keys() - images.keys())))
        depths = self._index(self.root / split / 'dense_depth') if self.input_channels == 4 else {}
        if self.input_channels == 4 and images.keys() != depths.keys():
            raise ValueError('%s depth IDs differ: missing depths=%s, extra depths=%s' %
                             (split, sorted(images.keys() - depths.keys()),
                              sorted(depths.keys() - images.keys())))
        for other in self.splits:
            folder = self.root / other / 'image_data'
            if other != split and folder.is_dir():
                overlap = images.keys() & self._index(folder).keys()
                if overlap:
                    raise ValueError('IDs overlap in %s/%s: %s' % (split, other, sorted(overlap)))
        self.sample_ids = sorted(images)
        self.samples = [(sid, images[sid], targets[sid], depths.get(sid)) for sid in self.sample_ids]

    @staticmethod
    def _index(folder):
        if not folder.is_dir():
            raise FileNotFoundError('Missing data directory: %s' % folder)
        files = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                sid = path.stem.casefold()
                if sid in files:
                    raise ValueError('Duplicate sample ID %s in %s' % (sid, folder))
                files[sid] = path
        if not files:
            raise ValueError('No image files in %s' % folder)
        return files

    @classmethod
    def encode_target(cls, target):
        raw = np.asarray(target)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('Raw labels must be a two-dimensional integer array')
        unexpected = set(np.unique(raw).tolist()) - set(cls.raw_to_train_id)
        if unexpected:
            raise ValueError('Unexpected raw label values: %s' % sorted(unexpected))
        encoded = np.empty(raw.shape, dtype=np.uint8)
        for raw_value, train_id in cls.raw_to_train_id.items():
            encoded[raw == raw_value] = train_id
        return encoded

    @classmethod
    def decode_target(cls, target):
        ids = np.asarray(target)
        if (not np.issubdtype(ids.dtype, np.integer) or
                np.any(ids < 0) or np.any(ids >= cls.num_classes)):
            raise ValueError('Class IDs must be integers in {0, 1, 2}')
        return cls.cmap[ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, image_path, target_path, depth_path = self.samples[index]
        try:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
            with Image.open(target_path) as source:
                target = Image.fromarray(self.encode_target(np.array(source)))
            if image.size != target.size:
                raise ValueError('Image WH %s and label WH %s differ' % (image.size, target.size))
            depth = read_depth(depth_path) if depth_path else None
            if depth is not None and depth.shape != (image.height, image.width):
                raise ValueError('Depth HW %s and RGB WH %s differ' % (depth.shape, image.size))
            if self.transform is None:
                return (image, target, depth) if depth is not None else (image, target)
            result = self.transform(image, target, return_metadata=self.return_metadata, depth=depth)
            if self.return_metadata:
                result[2]['sample_id'] = sample_id
            return result
        except (ValueError, OSError) as error:
            raise ValueError('%s / %s / %s: %s' % (image_path, target_path, depth_path, error)) from error
