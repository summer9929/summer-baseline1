"""odin1 image inference using the validation pipeline declared in main.py."""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from datasets import Odin1
from main import available_models, build_model, get_preprocess_config, get_transforms, load_checkpoint
from utils.seg_visualization import make_visuals, save_sample_results
from utils.preprocessing import read_depth


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='RGB image or image_data directory')
    parser.add_argument('--depth_input', default=None,
                        help='Paired 16-bit depth PNG or directory; defaults to sibling dense_depth')
    parser.add_argument('--input_channels', type=int, choices=[3, 4], default=4)
    parser.add_argument('--dataset', choices=['odin1'], default='odin1')
    parser.add_argument('--model', choices=available_models(), default='deeplabv3plus_mobilenet')
    parser.add_argument('--output_stride', type=int, choices=[8, 16], default=16)
    parser.add_argument('--separable_conv', action='store_true')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--save_val_results_to', default=None, help='Output directory; omit to infer without saving')
    return parser


def collect_image_files(input_path, output_dir=None):
    path = Path(input_path).expanduser().resolve()
    blocked = {'gt_image', 'dense_depth'}
    if any(part.casefold() in blocked for part in path.parts):
        raise ValueError('Specify RGB images, not gt_image or dense_depth: %s' % path)
    if path.is_file():
        return [(path, path.stem)]
    if not path.is_dir():
        raise FileNotFoundError('Input not found: %s' % path)
    if (path / 'image_data').is_dir():
        path = path / 'image_data'
    elif any((path / split).is_dir() for split in Odin1.splits):
        raise ValueError('Specify a split/image_data directory, not the dataset root: %s' % path)
    output = Path(output_dir).expanduser().resolve() if output_dir else None
    files, seen = [], set()
    for image_path in sorted(path.rglob('*')):
        relative = image_path.relative_to(path)
        if (not image_path.is_file() or image_path.suffix.lower() not in ('.png', '.jpg', '.jpeg')
                or any(part.casefold() in blocked for part in relative.parts)
                or (output is not None and image_path.is_relative_to(output))):
            continue
        sample_id = relative.with_suffix('').as_posix()
        if sample_id.casefold() in seen:
            raise ValueError('Output name collision: %s' % sample_id)
        seen.add(sample_id.casefold())
        files.append((image_path, sample_id))
    if not files:
        raise ValueError('No RGB images found: %s' % path)
    return files


def pair_depth_files(image_files, depth_input=None):
    """Match relative IDs; never infer missing measurements from an absent file."""
    if depth_input:
        root = Path(depth_input).expanduser().resolve()
        if root.is_file():
            if len(image_files) != 1:
                raise ValueError('A single depth file requires a single RGB image')
            return [root]
        if not root.is_dir():
            raise FileNotFoundError('Depth input not found: %s' % root)
        indexed = {}
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() == '.png':
                sid = path.relative_to(root).with_suffix('').as_posix().casefold()
                if sid in indexed:
                    raise ValueError('Duplicate depth ID: %s' % sid)
                indexed[sid] = path
        missing = [sid for _, sid in image_files if sid.casefold() not in indexed]
        if missing:
            raise ValueError('Missing paired depths: %s' % missing)
        return [indexed[sid.casefold()] for _, sid in image_files]
    depths = []
    for image_path, _ in image_files:
        rgb_root = next((p for p in image_path.parents if p.name.casefold() == 'image_data'), None)
        if rgb_root is None:
            raise ValueError('Cannot locate sibling dense_depth; specify --depth_input')
        depth_root = rgb_root.parent / 'dense_depth'
        relative = image_path.relative_to(rgb_root)
        files = Odin1._index(depth_root / relative.parent)
        if image_path.stem.casefold() not in files:
            raise ValueError('Missing paired depth for %s' % image_path)
        depths.append(files[image_path.stem.casefold()])
    return depths


def main():
    parser = get_argparser()
    opts = parser.parse_args()
    if not Path(opts.ckpt).is_file():
        parser.error('Checkpoint not found: %s' % opts.ckpt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_preprocess_config(opts.input_channels)
    _, transform = get_transforms(config)
    image_files = collect_image_files(opts.input, opts.save_val_results_to)
    depth_files = (pair_depth_files(image_files, opts.depth_input) if opts.input_channels == 4
                   else [None] * len(image_files))
    model = build_model(opts, pretrained_backbone=False)
    load_checkpoint(opts.ckpt, model, opts, config)
    model.to(device).eval()
    print('Device: %s; preprocessing: %s' % (device, config.to_dict()))
    with torch.no_grad():
        for (image_path, sample_id), depth_path in tqdm(list(zip(image_files, depth_files)), desc='odin1 inference'):
            try:
                with Image.open(image_path) as source:
                    image = source.convert('RGB')
                depth = read_depth(depth_path) if depth_path else None
                image, _, metadata = transform(image, return_metadata=True, depth=depth)
            except (ValueError, OSError) as error:
                raise ValueError('%s: %s' % (image_path, error)) from error
            pred = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            if opts.save_val_results_to:
                visuals = make_visuals(metadata['cropped_rgb'], pred, Odin1.decode_target)
                save_sample_results(opts.save_val_results_to, sample_id, visuals, Odin1.mask_gray_values)


if __name__ == '__main__':
    main()
