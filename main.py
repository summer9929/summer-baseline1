"""DeepLabV3+ odin1 training/evaluation and the single preprocessing declaration."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

import network
from network.input_fusion import expand_rgb_input, input_conv
import utils
from datasets import Odin1
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils.preprocessing import PreprocessConfig, SegmentationPreprocessor
from utils.seg_visualization import make_visuals, save_sample_results

PROJECT_ROOT = Path(__file__).resolve().parent
# All dimensions use (height, width). Change network input size ONLY here.
ODIN1_SOURCE_SIZE_HW = (1296, 1600)
ODIN1_CROP_SIZE_HW = (1088, 1280)
ODIN1_INPUT_SIZE_HW = (544, 640)
ODIN1_MEAN = (0.485, 0.456, 0.406)
ODIN1_STD = (0.229, 0.224, 0.225)
ODIN1_DEPTH_MEAN = 2.161799
ODIN1_DEPTH_STD = 2.206930


def get_preprocess_config(input_channels=4):
    return PreprocessConfig(ODIN1_SOURCE_SIZE_HW, ODIN1_CROP_SIZE_HW,
                            ODIN1_INPUT_SIZE_HW, ODIN1_MEAN, ODIN1_STD,
                            input_channels=input_channels, depth_mean=ODIN1_DEPTH_MEAN,
                            depth_std=ODIN1_DEPTH_STD)


def get_transforms(config=None):
    """Declare both deterministic pipelines here; predict.py uses val_transform."""
    config = config or get_preprocess_config()
    train_transform = SegmentationPreprocessor(
        config,
        crop_transform=et.ExtCenterCrop(config.crop_size_hw),
        input_transform=et.ExtCompose([
            et.ExtResize(config.input_size_hw),
            et.ExtToTensor(target_type='int64'),
            et.ExtNormalize(config.mean, config.std),
        ]),
    )
    val_transform = SegmentationPreprocessor(
        config,
        crop_transform=et.ExtCenterCrop(config.crop_size_hw),
        input_transform=et.ExtCompose([
            et.ExtResize(config.input_size_hw),
            et.ExtToTensor(target_type='int64'),
            et.ExtNormalize(config.mean, config.std),
        ]),
    )
    return train_transform, val_transform


def available_models():
    # Only advertise backbones whose implementation is present in this checkout.
    modules = {'mobilenet': 'mobilenetv2', 'resnet50': 'resnet', 'resnet101': 'resnet',
               'hrnetv2_32': 'hrnetv2', 'hrnetv2_48': 'hrnetv2', 'xception': 'xception'}
    return sorted(name for name, value in network.modeling.__dict__.items()
                  if name.startswith('deeplab') and callable(value)
                  and any(name.endswith('_' + suffix) and
                          (Path(network.__file__).parent / 'backbone' / (module + '.py')).is_file()
                          for suffix, module in modules.items()))


def get_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default=str(PROJECT_ROOT / 'segdataodin'))
    parser.add_argument('--dataset', choices=['odin1'], default='odin1')
    parser.add_argument('--num_classes', type=int, choices=[Odin1.num_classes], default=Odin1.num_classes)
    parser.add_argument('--model', default='deeplabv3plus_mobilenet', choices=available_models())
    parser.add_argument('--separable_conv', action='store_true')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16])
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--eval_split', choices=['validation', 'testing'], default='validation',
                        help='Split evaluated with --test_only; training always validates on validation')
    parser.add_argument('--save_val_results', action='store_true')
    parser.add_argument('--results_dir', default=str(PROJECT_ROOT / 'results'))
    parser.add_argument('--total_itrs', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_policy', choices=['poly', 'step'], default='poly')
    parser.add_argument('--step_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--input_channels', type=int, choices=[3, 4], default=4,
                        help='4: RGB-D baseline; 3: original RGB baseline')
    parser.add_argument('--init_rgb_ckpt', default=None,
                        help='Initialize RGB-D from an RGB three-class checkpoint; starts a fresh optimizer')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='Train from scratch without downloading backbone weights')
    parser.add_argument('--loss_type', choices=['cross_entropy', 'focal_loss'], default='cross_entropy')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--print_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--enable_vis', action='store_true')
    parser.add_argument('--vis_port', default='13570')
    parser.add_argument('--vis_env', default='odin1')
    parser.add_argument('--vis_num_samples', type=int, default=8)
    return parser


def get_dataset(opts, config=None):
    config = config or get_preprocess_config(opts.input_channels)
    train_transform, val_transform = get_transforms(config)
    keep_metadata = opts.save_val_results or opts.enable_vis
    eval_split = opts.eval_split if opts.test_only else 'validation'
    val_dst = Odin1(opts.data_root, eval_split, val_transform, keep_metadata)
    train_dst = None if opts.test_only else Odin1(opts.data_root, 'training', train_transform)
    return train_dst, val_dst


def build_model(opts, pretrained_backbone=False):
    model = network.modeling.__dict__[opts.model](
        num_classes=Odin1.num_classes, output_stride=opts.output_stride,
        pretrained_backbone=pretrained_backbone)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    expand_rgb_input(model, opts.input_channels)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    return model


def checkpoint_metadata(opts, config):
    return {
        'dataset': 'odin1', 'num_classes': Odin1.num_classes,
        'class_names': list(Odin1.class_names), 'raw_to_train_id': dict(Odin1.raw_to_train_id),
        'model': opts.model, 'output_stride': opts.output_stride,
        'separable_conv': opts.separable_conv, 'preprocess_config': config.to_dict(),
        'input_channels': config.input_channels,
        'channel_order': ['R_n', 'G_n', 'B_n', 'D_n'][:config.input_channels],
        'depth_encoding': ({'raw': 'floor(z_depth_m * 256)', 'axis': 'camera optical Z',
                            'invalid_raw': 0, 'saturated': False, 'numeric_clip': None,
                            'invalid_normalized': 0.0, 'interpolation': 'nearest',
                            'statistics': 'training valid pixels after center crop and resize'}
                           if config.input_channels == 4 else None),
    }


def load_checkpoint(path, model, opts, config):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: %s' % path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint:
        raise ValueError('Expected a project checkpoint with model_state: %s' % path)
    expected = checkpoint_metadata(opts, config)
    for key in ('dataset', 'num_classes', 'class_names', 'raw_to_train_id',
                'model', 'output_stride', 'separable_conv', 'input_channels', 'channel_order',
                'depth_encoding'):
        if key in checkpoint and checkpoint[key] != expected[key]:
            raise ValueError('Checkpoint %s mismatch: saved=%r, requested=%r' %
                             (key, checkpoint[key], expected[key]))
    if 'class_names' not in checkpoint or 'raw_to_train_id' not in checkpoint:
        print('Checkpoint lacks class metadata; class semantics cannot be verified automatically.')
    saved = checkpoint.get('preprocess_config')
    if config.input_channels == 4:
        if any(key not in checkpoint for key in ('input_channels', 'channel_order', 'depth_encoding')):
            raise ValueError('RGB-D checkpoint lacks input metadata; use --init_rgb_ckpt for RGB migration')
        if saved != expected['preprocess_config']:
            raise ValueError('RGB-D preprocessing mismatch; use the same geometry and depth statistics')
    if saved is None:
        print('Checkpoint lacks preprocessing metadata; using the configuration declared in main.py.')
    elif saved != expected['preprocess_config']:
        print('Preprocessing differs from training. Saved: %s; current: %s' %
              (json.dumps(saved), json.dumps(expected['preprocess_config'])))
    try:
        model.load_state_dict(checkpoint['model_state'], strict=True)
    except RuntimeError as error:
        raise ValueError('Checkpoint does not match the selected three-class model: %s' % error) from error
    print('Loaded checkpoint: %s' % path)
    return checkpoint


def initialize_from_rgb(path, model, opts):
    """Migrate only weights; RGB optimizer/scheduler and best score are never resumed."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint:
        raise ValueError('Expected a project RGB checkpoint with model_state')
    # The existing trained RGB file uses the older nested segdata_config schema.
    metadata = checkpoint.get('segdata_config', checkpoint)
    for key, expected in (('dataset', 'odin1'), ('model', opts.model),
                          ('num_classes', Odin1.num_classes), ('output_stride', opts.output_stride),
                          ('separable_conv', opts.separable_conv),
                          ('class_names', list(Odin1.class_names)),
                          ('raw_to_train_id', dict(Odin1.raw_to_train_id))):
        if key not in metadata or metadata[key] != expected:
            raise ValueError('RGB checkpoint %s mismatch or missing: %r' % (key, metadata.get(key)))
    state = dict(checkpoint['model_state'])
    name, conv = input_conv(model)
    key = name + '.weight'
    expected_shape = (conv.out_channels, 3, *conv.kernel_size)
    if key not in state or tuple(state[key].shape) != expected_shape or conv.in_channels != 4:
        raise ValueError('RGB migration requires a matching three-channel stem and a four-channel model')
    expanded = torch.zeros_like(conv.weight, device='cpu')
    expanded[:, :3].copy_(state[key])
    state[key] = expanded
    model.load_state_dict(state, strict=True)
    print('Initialized from RGB checkpoint (fresh optimizer, Depth weights zero): %s' % path)
    print('RGB source preprocessing: %s; target uses the current main.py configuration.' %
          metadata.get('preprocess_config', {k: metadata[k] for k in
                       ('input_size_hw', 'crop_box_ltrb', 'normalization') if k in metadata}))


def optimizer_groups(model, lr):
    raw = model.module if isinstance(model, nn.DataParallel) else model
    _, stem = input_conv(raw)
    if stem.in_channels == 3:
        return [{'params': list(raw.backbone.parameters()), 'lr': lr * 0.1},
                {'params': list(raw.classifier.parameters()), 'lr': lr}]
    stem_ids = {id(p) for p in stem.parameters()}
    return [
        {'params': [p for p in raw.backbone.parameters() if id(p) not in stem_ids], 'lr': lr * 0.1},
        {'params': list(stem.parameters()), 'lr': lr},
        {'params': list(raw.classifier.parameters()), 'lr': lr},
    ]


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Metrics use network resolution; result PNGs use the retained crop resolution."""
    metrics.reset()
    ret_samples = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=loader.dataset.split)):
            images, labels = batch[:2]
            outputs = model(images.to(device, dtype=torch.float32))
            if outputs.shape[1] != Odin1.num_classes or outputs.shape[-2:] != labels.shape[-2:]:
                raise ValueError('Model output and three-class label shape mismatch')
            preds = outputs.argmax(1).cpu().numpy()
            metrics.update(labels.numpy(), preds)
            selected = ret_samples_ids is not None and batch_index in ret_samples_ids
            if opts.save_val_results or selected:
                if len(batch) != 3:
                    raise ValueError('Full-size metadata is required for visualization')
                metadata = batch[2]
                for j, pred in enumerate(preds):
                    visuals = make_visuals(metadata['cropped_rgb'][j].numpy(), pred,
                                           loader.dataset.decode_target,
                                           metadata['cropped_target'][j].numpy())
                    sid = metadata['sample_id'][j]
                    if opts.save_val_results:
                        save_sample_results(Path(opts.results_dir) / loader.dataset.split, sid, visuals,
                                            loader.dataset.mask_gray_values)
                    if selected and j == 0:
                        ret_samples.append((sid, visuals))
    return metrics.get_results(), ret_samples


def report_score(score, config):
    h, w = config.input_size_hw
    print('Metric resolution WH: %dx%d' % (w, h))
    print(StreamSegMetrics.to_str(score))
    for class_id, name in enumerate(Odin1.class_names):
        print('Class %d (%s) IoU: %.6f' % (class_id, name, score['Class IoU'][class_id]))


def main():
    parser = get_argparser()
    opts = parser.parse_args()
    for key in ('batch_size', 'val_batch_size', 'total_itrs', 'step_size', 'print_interval', 'val_interval'):
        if getattr(opts, key) <= 0:
            parser.error('%s must be positive' % key)
    if opts.num_workers < 0 or opts.vis_num_samples < 0:
        parser.error('num_workers and vis_num_samples must be non-negative')
    if (opts.test_only or opts.continue_training) and not opts.ckpt:
        parser.error('--test_only and --continue_training require --ckpt')
    if opts.init_rgb_ckpt and (opts.ckpt or opts.test_only or opts.continue_training or opts.input_channels != 4):
        parser.error('--init_rgb_ckpt requires a fresh four-channel training run without --ckpt')
    if opts.init_rgb_ckpt and not Path(opts.init_rgb_ckpt).is_file():
        parser.error('RGB checkpoint not found: %s' % opts.init_rgb_ckpt)
    if opts.ckpt and not Path(opts.ckpt).is_file():
        parser.error('Checkpoint not found: %s' % opts.ckpt)
    if not opts.test_only and opts.batch_size < 2:
        parser.error('Training batch_size must be at least 2 for the existing ASPP BatchNorm')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    config = get_preprocess_config(opts.input_channels)
    print('Device: %s; preprocessing: %s' % (device, json.dumps(config.to_dict())))
    vis = None
    if opts.enable_vis:
        from utils.visualizer import Visualizer
        vis = Visualizer(port=opts.vis_port, env=opts.vis_env)
        vis.vis_table('Options', vars(opts))
    train_dst, val_dst = get_dataset(opts, config)
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False,
                                 num_workers=opts.num_workers)
    print('odin1 %s: %d samples' % (val_dst.split, len(val_dst)))
    model = build_model(opts, pretrained_backbone=(
        not opts.test_only and not opts.ckpt and not opts.init_rgb_ckpt and not opts.no_pretrained_backbone))
    if opts.init_rgb_ckpt:
        initialize_from_rgb(opts.init_rgb_ckpt, model, opts)
    checkpoint = load_checkpoint(opts.ckpt, model, opts, config) if opts.ckpt else None
    model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    metrics = StreamSegMetrics(Odin1.num_classes)
    sample_ids = set(range(min(opts.vis_num_samples, len(val_loader)))) if vis else None

    def evaluate():
        score, samples = validate(opts, model, val_loader, device, metrics, sample_ids)
        report_score(score, config)
        if vis:
            for sid, visuals in samples:
                for name in ('image', 'target_color', 'overlay'):
                    vis.vis_image('%s %s' % (sid, name), visuals[name].transpose(2, 0, 1))
        return score

    if opts.test_only:
        evaluate()
        return
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,
                                   num_workers=opts.num_workers, drop_last=True)
    if len(train_loader) == 0:
        parser.error('batch_size exceeds training sample count; reduce --batch_size')
    optimizer = torch.optim.SGD(optimizer_groups(model, opts.lr),
                                lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    scheduler = (utils.PolyLR(optimizer, opts.total_itrs, power=0.9) if opts.lr_policy == 'poly'
                 else torch.optim.lr_scheduler.StepLR(optimizer, opts.step_size, gamma=0.1))
    criterion = (utils.FocalLoss(ignore_index=255, size_average=True) if opts.loss_type == 'focal_loss'
                 else nn.CrossEntropyLoss(ignore_index=255))
    cur_itrs, cur_epochs, best_score = 0, 0, 0.0
    if opts.continue_training:
        for key in ('optimizer_state', 'scheduler_state', 'cur_itrs', 'best_score'):
            if key not in checkpoint:
                raise ValueError('Checkpoint cannot resume training: missing %s' % key)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        cur_itrs, best_score = int(checkpoint['cur_itrs']), float(checkpoint['best_score'])
        cur_epochs = int(checkpoint.get('cur_epochs', 0))
    del checkpoint
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    def save_ckpt(kind):
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        payload = checkpoint_metadata(opts, config)
        payload.update(cur_itrs=int(cur_itrs), cur_epochs=int(cur_epochs), best_score=float(best_score),
                       model_state=raw_model.state_dict(), optimizer_state=optimizer.state_dict(),
                       scheduler_state=scheduler.state_dict())
        if opts.init_rgb_ckpt:
            payload['rgb_initialization'] = str(Path(opts.init_rgb_ckpt).resolve())
        suffix = '_rgbd4' if config.input_channels == 4 else ''
        path = checkpoint_dir / ('%s_%s_odin1%s_os%d.pth' % (kind, opts.model, suffix, opts.output_stride))
        torch.save(payload, path)
        print('Model saved: %s' % path)

    interval_loss, interval_count = 0.0, 0
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device, dtype=torch.float32)),
                             labels.to(device, dtype=torch.long))
            loss.backward()
            optimizer.step()
            scheduler.step()
            cur_itrs += 1
            interval_loss += loss.item()
            interval_count += 1
            if vis:
                vis.vis_scalar('Loss', cur_itrs, loss.item())
            if cur_itrs % opts.print_interval == 0:
                print('Epoch %d, iteration %d/%d, loss %.6f' %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss / interval_count))
                interval_loss, interval_count = 0.0, 0
            if cur_itrs % opts.val_interval == 0:
                score = evaluate()
                if score['Mean IoU'] > best_score:
                    best_score = float(score['Mean IoU'])
                    save_ckpt('best')
                save_ckpt('latest')
                if vis:
                    vis.vis_scalar('[Val] Mean IoU', cur_itrs, score['Mean IoU'])
                    vis.vis_table('[Val] Class IoU', score['Class IoU'])
                model.train()
            if cur_itrs >= opts.total_itrs:
                break
    save_ckpt('latest')


if __name__ == '__main__':
    main()
