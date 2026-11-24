"""RGB-D contracts: geometry, encoding, migration, gradient and both CLI paths.

Random models and tiny injected models validate wiring, never segmentation accuracy.
"""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import main
import predict
from datasets import Odin1
from network.input_fusion import input_conv
from utils.preprocessing import normalize_depth, read_depth, resize_depth


class RGBDPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = main.get_preprocess_config()
        cls.root = main.PROJECT_ROOT / 'segdataodin'

    def test_decode_precision_invalid_and_no_clip(self):
        raw = np.array([[0, 1, 256, 65535]], dtype=np.uint16)
        got = normalize_depth(raw, self.config).numpy()[0]
        expected = np.zeros((1, 4), np.float32)
        expected[0, 1:] = (raw[0, 1:].astype(np.float32)/256 - 2.161799)/2.206930
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        self.assertEqual(got[0, 0], 0)
        self.assertGreater(got[0, 3], 100)  # 65535 is valid, not saturation or a clipped far depth.
        for invalid in (raw.astype(np.uint8), raw.astype(np.float32), raw[..., None]):
            with self.assertRaisesRegex(ValueError, 'uint16'):
                normalize_depth(invalid, self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'depth.png'
            Image.fromarray(raw).save(path)
            np.testing.assert_array_equal(read_depth(path), raw)

    def test_synchronized_geometry_and_all_three_stages(self):
        h, w = self.config.source_size_hw
        yy, xx = np.indices((h, w))
        rgb = Image.fromarray(np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8))
        label = Image.fromarray(((xx//31 + yy//17) % 3).astype(np.uint8))
        depth = ((xx + yy*3) % 65536).astype(np.uint16)
        train, val = main.get_transforms(self.config)
        a, ta = train(rgb, label, depth=depth)
        b, tb, meta = val(rgb, label, depth=depth, return_metadata=True)
        _, inference = predict.get_transforms(self.config)
        c, absent = inference(rgb, depth=depth)
        baseline = main.get_transforms(main.get_preprocess_config(3))[1](rgb, label)[0]
        self.assertEqual(tuple(a.shape), (4, 544, 640))
        self.assertTrue(torch.equal(a, b) and torch.equal(b, c))
        self.assertTrue(torch.equal(a[:3], baseline))
        self.assertTrue(torch.equal(ta, tb))
        self.assertIsNone(absent)
        expected = np.array(Image.fromarray(depth[104:1192, 160:1440]).resize(
            (640, 544), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(resize_depth(depth, self.config), expected)
        self.assertTrue(torch.equal(a[3:], normalize_depth(expected, self.config)))
        np.testing.assert_array_equal(meta['cropped_rgb'], np.asarray(rgb)[104:1192, 160:1440])
        with self.assertRaisesRegex(ValueError, 'requires a paired depth'):
            val(rgb, label)
        with self.assertRaisesRegex(ValueError, 'Depth source'):
            val(rgb, label, depth=np.zeros((12, 12), np.uint16))

    def test_real_samples_and_train_statistics(self):
        train_values = []
        for split, count in [('training', 18), ('validation', 6), ('testing', 1)]:
            dst = Odin1(self.root, split, main.get_transforms(self.config)[1])
            self.assertEqual(len(dst), count)
            for image, target in dst:
                self.assertEqual(tuple(image.shape), (4, 544, 640))
                self.assertTrue(torch.isfinite(image).all())
                self.assertEqual(target.dtype, torch.int64)
            if split == 'training':
                for _, _, _, path in dst.samples:
                    raw = resize_depth(read_depth(path), self.config)
                    train_values.append(raw[raw > 0].astype(np.float64)/256)
        values = np.concatenate(train_values)
        self.assertAlmostEqual(values.mean(), self.config.depth_mean, places=6)
        self.assertAlmostEqual(values.std(), self.config.depth_std, places=6)

    def test_missing_duplicate_and_invalid_depth_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for category in ['image_data', 'gt_image', 'dense_depth']:
                (root / 'testing' / category).mkdir(parents=True)
            Image.new('RGB', (1600, 1296)).save(root / 'testing/image_data/a.png')
            Image.new('L', (1600, 1296), 255).save(root / 'testing/gt_image/a.png')
            with self.assertRaisesRegex(ValueError, 'No image files'):
                Odin1(root, 'testing')
            Image.new('L', (1600, 1296)).save(root / 'testing/dense_depth/a.png')
            with self.assertRaisesRegex(ValueError, 'uint16'):
                Odin1(root, 'testing')[0]
            Image.fromarray(np.zeros((1296, 1600), np.uint16)).save(root / 'testing/dense_depth/a.png')
            dst = Odin1(root, 'testing', main.get_transforms(self.config)[1])
            image, target = dst[0]
            self.assertEqual(torch.count_nonzero(image[3]).item(), 0)
            self.assertEqual(target.unique().tolist(), [1])  # Missing depth still has supervised labels.
            Image.new('L', (1, 1)).save(root / 'testing/dense_depth/extra.png')
            with self.assertRaisesRegex(ValueError, 'depth IDs differ'):
                Odin1(root, 'testing')
            Image.new('L', (1, 1)).save(root / 'testing/dense_depth/a.jpg')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                Odin1(root, 'testing')

    def test_rgb_transfer_identity_depth_gradient_and_optimizer_groups(self):
        rgb_opts = main.get_argparser().parse_args(['--input_channels', '3'])
        opts = main.get_argparser().parse_args([])
        rgb = main.build_model(rgb_opts).eval()
        model = main.build_model(opts).eval()
        state = main.checkpoint_metadata(rgb_opts, main.get_preprocess_config(3))
        state['model_state'] = rgb.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rgb.pth'
            torch.save(state, path)
            main.initialize_from_rgb(path, model, opts)
            with self.assertRaisesRegex(ValueError, 'input_channels mismatch'):
                main.load_checkpoint(path, model, opts, self.config)
        name, conv = input_conv(model)
        self.assertEqual(tuple(conv.weight.shape), (32, 4, 3, 3))
        self.assertEqual(torch.count_nonzero(conv.weight[:, 3]).item(), 0)
        for key, original in rgb.state_dict().items():
            got = model.state_dict()[key]
            self.assertTrue(torch.equal(got[:, :3] if key == name+'.weight' else got, original))
        x = torch.randn(2, 4, 64, 80)
        with torch.no_grad():
            expected = rgb(x[:, :3])
        output = model(x)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        torch.nn.functional.cross_entropy(output, torch.randint(0, 3, (2, 64, 80))).backward()
        self.assertGreater(conv.weight.grad[:, 3].abs().sum().item(), 0)
        groups = main.optimizer_groups(model, 1e-3)
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual([g['lr'] for g in groups], [1e-4, 1e-3, 1e-3])

    def test_actual_rgb_checkpoint_migration(self):
        opts = main.get_argparser().parse_args([])
        model = main.build_model(opts).eval()
        main.initialize_from_rgb(self.root.parent / 'checkpoints/best_deeplabv3plus_mobilenet_odin1_os16.pth',
                                 model, opts)
        image, _ = Odin1(self.root, 'testing', main.get_transforms(self.config)[1])[0]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        self.assertEqual(tuple(output.shape), (1, 3, 544, 640))
        self.assertTrue(torch.isfinite(output).all())

    def test_native_checkpoint_and_both_cli_paths(self):
        opts = main.get_argparser().parse_args([])
        model = main.build_model(opts).eval()
        payload = main.checkpoint_metadata(opts, self.config)
        payload['model_state'] = model.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            ckpt = folder / 'synthetic_rgbd_smoke_only.pth'
            torch.save(payload, ckpt)
            main.load_checkpoint(ckpt, model, opts, self.config)
            for key in ['channel_order', 'depth_encoding', 'preprocess_config']:
                bad = deepcopy(payload)
                bad[key] = None
                torch.save(bad, folder / 'bad.pth')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    main.load_checkpoint(folder / 'bad.pth', model, opts, self.config)
            root = folder / 'data'
            shutil.copytree(self.root / 'testing', root / 'testing')
            images = predict.collect_image_files(root / 'testing')
            inferred = predict.pair_depth_files(images)
            explicit = predict.pair_depth_files(images, root / 'testing/dense_depth')
            self.assertEqual(inferred, explicit)
            self.assertEqual(predict.pair_depth_files(images, explicit[0]), explicit)
            with self.assertRaisesRegex(ValueError, 'Missing paired'):
                predict.pair_depth_files(images, folder)
            with patch.object(torch.cuda, 'is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['main.py', '--test_only', '--eval_split', 'testing',
                                       '--ckpt', str(ckpt), '--data_root', str(root), '--num_workers', '0',
                                       '--save_val_results', '--results_dir', str(folder / 'eval')]):
                    main.main()
                with patch('sys.argv', ['predict.py', '--input', str(root / 'testing/image_data'),
                                       '--ckpt', str(ckpt), '--save_val_results_to', str(folder / 'infer')]):
                    predict.main()
            eval_files = sorted((folder / 'eval/testing').glob('*.png'))
            infer_files = sorted((folder / 'infer').glob('*.png'))
            self.assertEqual(len(eval_files), 5)
            self.assertEqual(len(infer_files), 3)
            for path in infer_files:
                with Image.open(path) as a, Image.open(folder / 'eval/testing' / path.name) as b:
                    np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_training_defaults_preserved(self):
        opts = main.get_argparser().parse_args([])
        self.assertEqual((opts.lr, opts.total_itrs, opts.batch_size, opts.val_interval),
                         (0.01, 30000, 16, 100))
        self.assertEqual((opts.lr_policy, opts.weight_decay, opts.loss_type),
                         ('poly', 1e-4, 'cross_entropy'))

    def test_four_channel_training_save_and_resume_with_tiny_injected_model(self):
        class TinySeg(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1),
                                                    torch.nn.Conv2d(2, 2, 1))
                self.classifier = torch.nn.Conv2d(2, 3, 1)

            def forward(self, x):
                return self.classifier(self.backbone(x))

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            root = folder / 'data'
            for split, count in [('training', 2), ('validation', 1)]:
                for category in ['image_data', 'gt_image', 'dense_depth']:
                    dest = root / split / category
                    dest.mkdir(parents=True)
                    for path in sorted((self.root / split / category).glob('*.png'))[:count]:
                        shutil.copyfile(path, dest / path.name)
            args = ['main.py', '--data_root', str(root), '--batch_size', '2', '--val_batch_size', '1',
                    '--num_workers', '0', '--val_interval', '1', '--print_interval', '1',
                    '--no_pretrained_backbone', '--lr_policy', 'step']
            with patch.object(main, 'PROJECT_ROOT', folder), \
                    patch.object(main, 'build_model', side_effect=lambda *a, **kw: TinySeg()), \
                    patch.object(torch.cuda, 'is_available', return_value=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', args + ['--total_itrs', '1']):
                    main.main()
                ckpt = folder / 'checkpoints/latest_deeplabv3plus_mobilenet_odin1_rgbd4_os16.pth'
                first = torch.load(ckpt, weights_only=True)
                self.assertEqual(first['cur_itrs'], 1)
                self.assertEqual(first['input_channels'], 4)
                self.assertEqual(first['preprocess_config'], self.config.to_dict())
                self.assertEqual(len(first['optimizer_state']['param_groups']), 3)
                with patch('sys.argv', args + ['--total_itrs', '2', '--continue_training', '--ckpt', str(ckpt)]):
                    main.main()
                resumed = torch.load(ckpt, weights_only=True)
                self.assertEqual(resumed['cur_itrs'], 2)
                self.assertEqual(resumed['scheduler_state']['last_epoch'], 2)
                for tensor in resumed['model_state'].values():
                    self.assertTrue(torch.isfinite(tensor).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
