"""Contract checks with unittest; synthetic checkpoints are NOT accuracy evaluation."""

import contextlib
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

import calibseg
from datasets import Odin1
import main
from metrics import StreamSegMetrics
import predict
from utils import ext_transforms as et
from utils.preprocessing import get_geometry, restore_mask
from utils.seg_visualization import make_visuals, save_sample_results


class Odin1PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = main.get_preprocess_config(3)
        cls.train_transform, cls.val_transform = main.get_transforms(cls.config)
        cls.root = main.PROJECT_ROOT / 'segdataodin'

    def test_all_real_samples(self):
        counts, all_ids = {}, set()
        for split in Odin1.splits:
            dataset = Odin1(self.root, split, self.val_transform)
            counts[split] = len(dataset)
            self.assertFalse(all_ids.intersection(dataset.sample_ids))
            all_ids.update(dataset.sample_ids)
            for image, target in dataset:
                self.assertEqual(tuple(image.shape), (3, *self.config.input_size_hw))
                self.assertEqual(tuple(target.shape), self.config.input_size_hw)
                self.assertEqual(image.dtype, torch.float32)
                self.assertEqual(target.dtype, torch.int64)
                self.assertTrue(torch.isfinite(image).all())
                self.assertLessEqual(set(target.unique().tolist()), {0, 1, 2})
        self.assertEqual(counts, {'training': 18, 'validation': 6, 'testing': 1})

    def test_raw_label_mapping_and_rejection(self):
        raw = np.array([[0, 255, 128]], dtype=np.uint8)
        np.testing.assert_array_equal(Odin1.encode_target(raw), [[0, 1, 2]])
        np.testing.assert_array_equal(raw, [[0, 255, 128]])
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            Odin1.encode_target(np.array([[42]], dtype=np.uint8))
        np.testing.assert_array_equal(Odin1.decode_target(np.array([0, 1, 2])),
                                      [[220, 50, 50], [40, 200, 60], [60, 140, 240]])

    def test_crop_geometry_and_three_stage_tensor_identity(self):
        h, w = self.config.source_size_hw
        yy, xx = np.indices((h, w))
        rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), -1).astype(np.uint8)
        target = ((xx // 50 + yy // 40) % 3).astype(np.uint8)
        pil, label = Image.fromarray(rgb), Image.fromarray(target)
        train_image, train_target = self.train_transform(pil, label)
        val_image, val_target, metadata = self.val_transform(pil, label, return_metadata=True)
        _, inference = predict.get_transforms(predict.get_preprocess_config(3))
        infer_image, absent, context = inference(pil, return_metadata=True)
        self.assertIsNone(absent)
        self.assertTrue(torch.equal(train_image, val_image))
        self.assertTrue(torch.equal(val_image, infer_image))
        self.assertTrue(torch.equal(train_target, val_target))
        np.testing.assert_array_equal(metadata['geometry']['crop_box_ltrb'], [160, 104, 1440, 1192])
        np.testing.assert_array_equal(metadata['cropped_rgb'], rgb[104:1192, 160:1440])
        np.testing.assert_array_equal(context['cropped_rgb'], metadata['cropped_rgb'])
        np.testing.assert_array_equal(metadata['cropped_target'], target[104:1192, 160:1440])
        expected = Image.fromarray(target[104:1192, 160:1440]).resize((640, 544), Image.Resampling.NEAREST)
        np.testing.assert_array_equal(val_target.numpy(), np.array(expected))
        with self.assertRaisesRegex(ValueError, 'source WH'):
            inference(Image.new('RGB', (1000, 1000)))

    def test_input_size_changes_only_in_main(self):
        with patch.object(main, 'ODIN1_INPUT_SIZE_HW', (672, 800)):
            config = predict.get_preprocess_config(3)
            _, transform = predict.get_transforms(config)
            tensor, _, metadata = transform(Image.new('RGB', (1600, 1296)), return_metadata=True)
            self.assertEqual(tuple(tensor.shape), (3, 672, 800))
            self.assertEqual(metadata['cropped_rgb'].shape, (1088, 1280, 3))
            self.assertEqual(restore_mask(np.zeros((672, 800), np.uint8), config.crop_size_hw).shape,
                             (1088, 1280))

    def test_padding_preserves_classes_and_uses_ignore(self):
        image = Image.new('RGB', (5, 3), (100, 100, 100))
        label = Image.fromarray(np.ones((3, 5), np.uint8))
        padded, target = et.ExtPad(diviser=4)(image, label)
        self.assertEqual(padded.size, (8, 4))
        np.testing.assert_array_equal(np.array(target)[:3, 1:6], np.ones((3, 5), np.uint8))
        self.assertEqual(np.array(target)[-1, -1], 255)
        self.assertEqual(et.ExtPad(diviser=4)(image)[0].size, (8, 4))

    def test_visualization_formula_restoration_and_five_files(self):
        pred = np.tile(np.array([0, 1, 2], np.uint8), (2, 1))
        image = np.full((4, 6, 3), 100, np.uint8)
        target = np.repeat(np.repeat(pred, 2, 0), 2, 1)
        visuals = make_visuals(image, pred, Odin1.decode_target, target)
        self.assertEqual(set(visuals), {'image', 'pred_id', 'overlay', 'target_id', 'target_color'})
        np.testing.assert_array_equal(visuals['pred_id'], target)
        np.testing.assert_array_equal(visuals['overlay'][0, ::2],
                                      [[148, 80, 80], [76, 140, 84], [84, 116, 156]])
        with tempfile.TemporaryDirectory() as folder:
            save_sample_results(folder, 'sample', visuals, Odin1.mask_gray_values)
            self.assertEqual(len(list(Path(folder).glob('*.png'))), 5)
            self.assertFalse((Path(folder) / 'sample_pred_color.png').exists())
            for path in Path(folder).glob('*.png'):
                with Image.open(path) as saved:
                    self.assertEqual(saved.size, (6, 4))
                    np.testing.assert_array_equal(np.array(saved), visuals[path.stem[7:]])
                    if path.stem in ('sample_pred_id', 'sample_target_id'):
                        self.assertEqual(saved.mode, 'P')
                        np.testing.assert_array_equal(
                            np.array(saved.convert('L')),
                            np.array([[0, 0, 255, 255, 128, 128]] * 4, dtype=np.uint8))

    def test_calibration_projection_and_original_preservation(self):
        source = self.root / 'calib.yaml'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        original = yaml.safe_load(source.read_text(encoding='utf-8'))
        crop = calibseg.derive_calibration(original, self.config)
        net = calibseg.derive_calibration(original, self.config, True)
        self.assertAlmostEqual(crop['cam_0']['u0'], 656.1257953104363)
        self.assertAlmostEqual(crop['cam_0']['v0'], 553.6595263441989)
        for derived, shift, scale in ((crop, (160, 104), 1.0), (net, (160, 104), 0.5)):
            self.assertEqual(derived['Tcl_0'], original['Tcl_0'])
            for key in ('k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'p1', 'p2'):
                self.assertEqual(derived['cam_0'][key], original['cam_0'][key])
            for x, y in ((0., 0.), (.2, -.3), (-.4, .5)):
                a, b = original['cam_0'], derived['cam_0']
                raw = np.array([a['A11']*x + a['A12']*y + a['u0'], a['A22']*y + a['v0']])
                expected = (raw - shift + .5) * scale - .5
                actual = [b['A11']*x + b['A12']*y + b['u0'], b['A22']*y + b['v0']]
                np.testing.assert_allclose(actual, expected, atol=1e-10)
        with tempfile.TemporaryDirectory() as folder:
            paths = calibseg.export_calibrations(source, folder, self.config)
            self.assertEqual([p.name for p in paths], ['calib_1280x1088.yaml', 'calib_640x544.yaml'])
            self.assertEqual(yaml.safe_load(paths[1].read_text())['cam_0']['image_width'], 640)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_pairing_and_depth_is_optional(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('image_data', 'gt_image'):
                (root / 'testing' / name).mkdir(parents=True)
            Image.new('RGB', (1600, 1296)).save(root / 'testing/image_data/a.png')
            with self.assertRaisesRegex(ValueError, 'No image files'):
                Odin1(root, 'testing', self.val_transform)
            Image.new('L', (1600, 1296), 255).save(root / 'testing/gt_image/a.png')
            dataset = Odin1(root, 'testing', self.val_transform)
            self.assertEqual(dataset[0][1].unique().tolist(), [1])
            Image.new('RGB', (1600, 1296)).save(root / 'testing/image_data/a.jpg')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                Odin1(root, 'testing', self.val_transform)

    def test_eval_split_and_no_training_dataset(self):
        opts = main.get_argparser().parse_args(['--test_only', '--eval_split', 'testing', '--input_channels', '3'])
        train, test = main.get_dataset(opts, self.config)
        self.assertIsNone(train)
        self.assertEqual(test.split, 'testing')
        opts.test_only = False
        train, val = main.get_dataset(opts, self.config)
        self.assertEqual(train.split, 'training')
        self.assertEqual(val.split, 'validation')

    def test_prediction_directory_guards(self):
        with self.assertRaisesRegex(ValueError, 'RGB'):
            predict.collect_image_files(self.root / 'testing/gt_image')
        with self.assertRaisesRegex(ValueError, 'dataset root'):
            predict.collect_image_files(self.root)
        self.assertEqual(len(predict.collect_image_files(self.root / 'testing')), 1)

    def test_windows_workers_and_full_size_metadata(self):
        dataset = Odin1(self.root, 'validation', self.val_transform, return_metadata=True)
        loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)
        sample_ids = []
        for images, labels, metadata in loader:
            self.assertEqual(tuple(images.shape[1:]), (3, 544, 640))
            self.assertEqual(labels.dtype, torch.int64)
            self.assertEqual(tuple(metadata['cropped_rgb'].shape[1:]), (1088, 1280, 3))
            self.assertEqual(tuple(metadata['geometry']['crop_box_ltrb'].shape[1:]), (4,))
            sample_ids.extend(metadata['sample_id'])
        self.assertEqual(sample_ids, dataset.sample_ids)

    def test_training_loop_checkpoint_and_resume(self):
        # A tiny injected model checks optimizer/scheduler/resume wiring without a real training run.
        class TinySeg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Conv2d(3, 2, 1)
                self.classifier = torch.nn.Conv2d(2, 3, 1)

            def forward(self, images):
                return self.classifier(self.backbone(images))

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            root = folder / 'data'
            for split, count in (('training', 2), ('validation', 1)):
                for category in ('image_data', 'gt_image'):
                    dest = root / split / category
                    dest.mkdir(parents=True)
                    for path in sorted((self.root / split / category).glob('*.png'))[:count]:
                        shutil.copyfile(path, dest / path.name)
            args = ['main.py', '--input_channels', '3', '--data_root', str(root), '--batch_size', '2',
                    '--val_batch_size', '1', '--num_workers', '0', '--val_interval', '1',
                    '--print_interval', '1', '--no_pretrained_backbone', '--lr_policy', 'step']
            with patch.object(main, 'PROJECT_ROOT', folder), \
                    patch.object(main, 'build_model', side_effect=lambda *a, **kw: TinySeg()), \
                    patch.object(torch.cuda, 'is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', args + ['--total_itrs', '1']):
                    main.main()
                ckpt = folder / 'checkpoints/latest_deeplabv3plus_mobilenet_odin1_os16.pth'
                first = torch.load(ckpt, weights_only=True)
                self.assertEqual(first['cur_itrs'], 1)
                self.assertIs(type(first['best_score']), float)
                self.assertEqual(first['scheduler_state']['last_epoch'], 1)
                self.assertEqual(first['preprocess_config'], self.config.to_dict())
                self.assertTrue(first['optimizer_state']['state'])
                with patch('sys.argv', args + ['--total_itrs', '2', '--continue_training', '--ckpt', str(ckpt)]):
                    main.main()
                resumed = torch.load(ckpt, weights_only=True)
                self.assertEqual(resumed['cur_itrs'], 2)
                self.assertEqual(resumed['scheduler_state']['last_epoch'], 2)
                self.assertGreater(resumed['cur_epochs'], first['cur_epochs'])
                for value in resumed['model_state'].values():
                    self.assertTrue(torch.isfinite(value).all())

    def test_validate_metrics_and_metadata(self):
        dataset = Odin1(self.root, 'testing', self.val_transform, return_metadata=True)
        labels = dataset[0][1]
        class KnownLabels(torch.nn.Module):
            def forward(self, images):
                return torch.nn.functional.one_hot(labels, 3).permute(2, 0, 1).float().unsqueeze(0)
        with tempfile.TemporaryDirectory() as folder:
            opts = main.get_argparser().parse_args(['--save_val_results', '--results_dir', folder])
            score, samples = main.validate(opts, KnownLabels(), DataLoader(dataset), 'cpu',
                                            StreamSegMetrics(3), {0})
            self.assertAlmostEqual(score['Mean IoU'], 1.0)
            self.assertEqual(samples[0][0], dataset.sample_ids[0])
            for path in (Path(folder) / 'testing').glob('*.png'):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1280, 1088))

    def test_real_model_and_both_cli_paths_with_synthetic_checkpoint(self):
        # Random initialization verifies wiring and shape, NOT segmentation quality.
        opts = main.get_argparser().parse_args(['--input_channels', '3'])
        torch.manual_seed(1)
        model = main.build_model(opts, pretrained_backbone=False).eval()
        payload = main.checkpoint_metadata(opts, self.config)
        payload['model_state'] = model.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            ckpt = folder / 'synthetic_smoke_only.pth'
            torch.save(payload, ckpt)
            with contextlib.redirect_stdout(io.StringIO()):
                main.load_checkpoint(ckpt, model, opts, self.config)
            bad = deepcopy(payload)
            bad['class_names'] = ['wrong', 'wrong', 'wrong']
            torch.save(bad, folder / 'bad.pth')
            with self.assertRaisesRegex(ValueError, 'class_names mismatch'):
                main.load_checkpoint(folder / 'bad.pth', model, opts, self.config)
            sample = Odin1(self.root, 'testing', self.val_transform)[0][0]
            with torch.no_grad():
                output = model(sample.unsqueeze(0))
            self.assertEqual(tuple(output.shape), (1, 3, 544, 640))
            self.assertTrue(torch.isfinite(output).all())
            # CLI evaluation must work with only the testing tree present.
            tiny_root = folder / 'data'
            shutil.copytree(self.root / 'testing', tiny_root / 'testing',
                            ignore=shutil.ignore_patterns('dense_depth'))
            with patch.object(torch.cuda, 'is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['main.py', '--input_channels', '3', '--test_only', '--eval_split', 'testing',
                                       '--ckpt', str(ckpt), '--data_root', str(tiny_root),
                                       '--num_workers', '0', '--save_val_results',
                                       '--results_dir', str(folder / 'eval')]):
                    main.main()
                with patch('sys.argv', ['predict.py', '--input_channels', '3', '--input', str(tiny_root / 'testing/image_data'),
                                       '--ckpt', str(ckpt), '--save_val_results_to', str(folder / 'infer')]):
                    predict.main()
            eval_files = sorted((folder / 'eval/testing').glob('*.png'))
            infer_files = sorted((folder / 'infer').glob('*.png'))
            self.assertEqual(len(eval_files), 5)
            self.assertEqual(len(infer_files), 3)
            for path in infer_files:
                with Image.open(path) as a, Image.open(folder / 'eval/testing' / path.name) as b:
                    self.assertEqual(a.size, (1280, 1088))
                    np.testing.assert_array_equal(np.array(a), np.array(b))
            for argv in (['main.py', '--test_only'], ['predict.py', '--input', str(tiny_root)]):
                with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        main.main() if argv[0] == 'main.py' else predict.main()
                    self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
