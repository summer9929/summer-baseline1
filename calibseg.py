"""Derive FishPoly calibration for main.py's crop and network sizes; preserve the raw file."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from main import PROJECT_ROOT, get_preprocess_config
from utils.preprocessing import get_geometry


def derive_calibration(calibration, config, network_input=False):
    camera = calibration['cam_0']
    if calibration.get('cam_num') != 1 or camera.get('cam_model') != 'FishPoly':
        raise ValueError('Expected the single-camera odin1 FishPoly calibration')
    if (camera['image_height'], camera['image_width']) != config.source_size_hw:
        raise ValueError('Calibration source dimensions differ from main.py')
    geometry = get_geometry(config)
    left, top, _, _ = geometry['crop_box_ltrb'].tolist()
    crop_h, crop_w = config.crop_size_hw
    height, width = config.input_size_hw if network_input else config.crop_size_hw
    sx, sy = width / crop_w, height / crop_h
    result = deepcopy(calibration)
    transformed = result['cam_0']
    transformed.update(image_width=width, image_height=height)
    # Integer pixel centers; Pillow resize maps x' = (x + .5) * scale - .5.
    transformed['u0'] = (camera['u0'] - left + 0.5) * sx - 0.5
    transformed['v0'] = (camera['v0'] - top + 0.5) * sy - 0.5
    transformed['A11'] = camera['A11'] * sx
    transformed['A12'] = camera['A12'] * sx
    transformed['A22'] = camera['A22'] * sy
    result['segmentation_preprocessing'] = {
        **{key: value.tolist() for key, value in geometry.items()},
        'output_size_hw': [height, width],
        'pixel_convention': 'integer centers; resize: (pixel + 0.5) * scale - 0.5',
        'error_statistics_note': 'rms_error_px and median_error_px are original calibration statistics, not refitted',
    }
    return result


def export_calibrations(source, output_dir, config=None):
    source = Path(source).expanduser().resolve()
    config = config or get_preprocess_config()
    with source.open(encoding='utf-8') as handle:
        calibration = yaml.safe_load(handle)
    output_dir = Path(output_dir).expanduser().resolve()
    paths = []
    for size_hw in (config.crop_size_hw, config.input_size_hw):
        height, width = size_hw
        path = output_dir / ('calib_%dx%d.yaml' % (width, height))
        if path == source:
            raise ValueError('Derived calibration must not overwrite the original')
        paths.append(path)
    # Build both before writing, so malformed input cannot leave a half-derived pair.
    payloads = [derive_calibration(calibration, config, False),
                derive_calibration(calibration, config, True)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(paths, payloads):
        payload['segmentation_preprocessing']['source_calibration'] = str(source)
        with path.open('w', encoding='utf-8') as handle:
            handle.write('# Derived calibration; original error statistics are retained for provenance.\n')
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calib', default=str(PROJECT_ROOT / 'segdataodin' / 'calib.yaml'))
    parser.add_argument('--output_dir', default=None)
    opts = parser.parse_args()
    output_dir = opts.output_dir or Path(opts.calib).expanduser().resolve().parent
    for path in export_calibrations(opts.calib, output_dir):
        print('Saved: %s' % path)


if __name__ == '__main__':
    main()
