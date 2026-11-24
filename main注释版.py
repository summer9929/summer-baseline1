"""DeepLabV3+ odin1 training/evaluation and the single preprocessing declaration."""

# =============================================================================
# main.py —— DeepLabV3+ 三分类（不可通行 / 可通行 / 天空）训练与评估入口
# =============================================================================
# 设计要点：
#   1. "预处理只有一个声明处"：本文件顶部的 4 个尺寸常量 + mean/std 就是全项目
#      唯一的尺寸/归一化真源，其他模块（Dataset、predict）都从这里取。
#   2. 数据管线（H,W 约定）：
#         原图 1296×1600
#           → 中心裁剪 1088×1280     （ODIN1_CROP_SIZE_HW）
#           → 缩放     544×640       （ODIN1_INPUT_SIZE_HW，即网络输入）
#           → ImageNet 归一化
#      指标在 544×640 上统计；保存的结果图会最近邻还原回 1088×1280。
#   3. 训练与验证使用**完全相同**的 transform（无任何随机增强）。
#   4. 检查点内嵌元数据（类别语义 + 预处理配置），加载时做一致性校验。
# =============================================================================

import argparse                 # 命令行参数
import json                     # 打印/保存预处理配置为 JSON
import os                       # 环境变量（CUDA_VISIBLE_DEVICES）
from pathlib import Path        # 路径处理，PROJECT_ROOT 定位
import random                   # Python 随机种子

import numpy as np              # 随机种子、数组操作
import torch
from torch import nn            # CrossEntropyLoss / DataParallel
from torch.utils import data    # DataLoader
from tqdm import tqdm           # 验证进度条

import network                  # 模型工厂 network.modeling
import utils                    # PolyLR / FocalLoss / set_bn_momentum
from datasets import Odin1      # odin1 三分类数据集
from metrics import StreamSegMetrics          # 混淆矩阵 → mIoU/Acc/各类 IoU
from utils import ext_transforms as et        # 成对几何/张量变换（Resize/CenterCrop/ToTensor/Normalize）
from utils.preprocessing import PreprocessConfig, SegmentationPreprocessor
# PreprocessConfig        ：不可变的预处理配置（源尺寸/裁剪尺寸/输入尺寸/均值/标准差）
# SegmentationPreprocessor：把"裁剪 → 缩放/归一化"两级串起来的可调用对象
from utils.seg_visualization import make_visuals, save_sample_results
# make_visuals        ：生成 image / pred_id / overlay / target_* 等可视化数组
# save_sample_results ：按 <样本ID>_<类型>.png 落盘

# 项目根目录（本文件所在目录），后续 checkpoints/results 都以此为基准
PROJECT_ROOT = Path(__file__).resolve().parent
# All dimensions use (height, width). Change network input size ONLY here.
# ---- 全项目唯一的尺寸声明处：注意全部是 (高, 宽) 顺序 ----
ODIN1_SOURCE_SIZE_HW = (1296, 1600)   # 原始图像尺寸（DTOF 模组输出）
ODIN1_CROP_SIZE_HW = (1088, 1280)     # 中心裁剪后的尺寸（上下各去 104 行、左右各去 160 列）
ODIN1_INPUT_SIZE_HW = (544, 640)      # 网络实际输入尺寸（恰为裁剪尺寸的一半）
# ImageNet 归一化参数（与预训练主干一致）
ODIN1_MEAN = (0.485, 0.456, 0.406)
ODIN1_STD = (0.229, 0.224, 0.225)


def get_preprocess_config():
    """把上面的常量组装成不可变的配置对象（构造时会做合法性校验）。"""
    # PreprocessConfig.__post_init__ 会校验：尺寸为正整数二元组、裁剪不超过源尺寸、std 全为正
    return PreprocessConfig(ODIN1_SOURCE_SIZE_HW, ODIN1_CROP_SIZE_HW,
                            ODIN1_INPUT_SIZE_HW, ODIN1_MEAN, ODIN1_STD)


def get_transforms(config=None):
    """Declare both deterministic pipelines here; predict.py uses val_transform."""
    # 未显式传入配置时，使用全局声明的那一份
    config = config or get_preprocess_config()
    # 训练管线：中心裁剪 → 缩放 → 转张量 → 归一化
    # 注意：这里没有任何随机增强（没有 RandomCrop / Flip / ColorJitter）
    train_transform = SegmentationPreprocessor(
        config,
        # 第一级：按 (H,W) 做中心裁剪，图像与标签同框
        crop_transform=et.ExtCenterCrop(config.crop_size_hw),
        # 第二级：精确缩放到网络输入尺寸，然后转张量并归一化
        # ExtToTensor(target_type='int64')：标签保持整数类别，不做 [0,1] 缩放
        input_transform=et.ExtCompose([
            et.ExtResize(config.input_size_hw),
            et.ExtToTensor(target_type='int64'),
            et.ExtNormalize(config.mean, config.std),
        ]),
    )
    # 验证管线：与训练管线**逐行相同**（确定性，无增强）
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
    """列出 network.modeling 中所有以 'deeplab' 开头的可调用模型构造器。"""
    # 用于 --model 的 choices，避免手动维护模型名列表
    return sorted(name for name, value in network.modeling.__dict__.items()
                  if name.startswith('deeplab') and callable(value))


def get_argparser():
    """声明全部命令行参数（真正的取值校验在 main() 开头）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- 数据相关 ----
    # 默认指向项目下的 segdataodin（绝对路径，不受当前工作目录影响）
    parser.add_argument('--data_root', default=str(PROJECT_ROOT / 'segdataodin'))
    # 目前只支持 odin1（单数据集版本，不含混合模式）
    parser.add_argument('--dataset', choices=['odin1'], default='odin1')
    # 类别数由 Odin1.num_classes 唯一确定（=3），传其他值会被 argparse 直接拒绝
    parser.add_argument('--num_classes', type=int, choices=[Odin1.num_classes], default=Odin1.num_classes)
    # ---- 模型相关 ----
    parser.add_argument('--model', default='deeplabv3plus_mobilenet', choices=available_models())
    # 把 decoder/ASPP 的 3×3 卷积换成深度可分离卷积（会改变权重键，需与 ckpt 一致）
    parser.add_argument('--separable_conv', action='store_true')
    # 16 → 特征图 1/16、ASPP 膨胀率 [6,12,18]；8 → 1/8、[12,24,36]
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16])
    # ---- 运行模式 ----
    # 只评估，不训练
    parser.add_argument('--test_only', action='store_true')
    # 评估用哪个划分：训练时固定用 validation，只有 test_only 才能选 testing
    parser.add_argument('--eval_split', choices=['validation', 'testing'], default='validation',
                        help='Split evaluated with --test_only; training always validates on validation')
    # 保存结果图（image/pred_id/overlay/target_* 等）
    parser.add_argument('--save_val_results', action='store_true')
    # 结果根目录；实际写入 results/<split>/<sample_id>_<类型>.png
    parser.add_argument('--results_dir', default=str(PROJECT_ROOT / 'results'))
    # ---- 训练预算 ----
    # 优化器更新总次数（不是 epoch）
    parser.add_argument('--total_itrs', type=int, default=30000)
    # 基础学习率：主干用 0.1×，分类头用 1×
    parser.add_argument('--lr', type=float, default=0.01)
    # poly → PolyLR(total_itrs, power=0.9)；step → StepLR(step_size, gamma=0.1)
    parser.add_argument('--lr_policy', choices=['poly', 'step'], default='poly')
    parser.add_argument('--step_size', type=int, default=10000)
    # 批大小：训练必须 ≥2（ASPP 里的 BatchNorm 约束），main() 会检查
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    # 注意：这里用的是 --num_workers（不是 --workers）
    parser.add_argument('--num_workers', type=int, default=2)
    # ---- 权重 ----
    # ckpt：完整检查点（含元数据），用于评估或续训
    parser.add_argument('--ckpt', default=None)
    # 续训：恢复优化器/调度器/迭代数/最佳分数
    parser.add_argument('--continue_training', action='store_true')
    # 从零训练：不下载 ImageNet 主干权重（主干随机初始化）
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='Train from scratch without downloading backbone weights')
    # ---- 其它 ----
    # 损失函数：focal_loss 的 gamma 默认为 0，实际近似退化为交叉熵
    parser.add_argument('--loss_type', choices=['cross_entropy', 'focal_loss'], default='cross_entropy')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=1)
    # 日志打印间隔（本文件中确实被使用）
    parser.add_argument('--print_interval', type=int, default=10)
    # 每多少次更新验证一次（并触发保存）
    parser.add_argument('--val_interval', type=int, default=100)
    # ---- visdom 可视化（可选） ----
    parser.add_argument('--enable_vis', action='store_true')
    parser.add_argument('--vis_port', default='13570')
    parser.add_argument('--vis_env', default='odin1')
    parser.add_argument('--vis_num_samples', type=int, default=8)
    return parser


def get_dataset(opts, config=None):
    """构建 (train_dst, val_dst)；test_only 时 train_dst 为 None。"""
    train_transform, val_transform = get_transforms(config)
    # 是否保留"裁剪后但未缩放"的全尺寸像素与元数据：
    #   保存结果图或 visdom 可视化时需要（否则无法按 1088×1280 出图）
    keep_metadata = opts.save_val_results or opts.enable_vis
    # 评估划分：仅 test_only 时可切到 testing；训练/常规验证固定 validation
    eval_split = opts.eval_split if opts.test_only else 'validation'
    # 验证集（或测试集）：传入 val_transform；keep_metadata 控制返回 2 元组还是 3 元组
    val_dst = Odin1(opts.data_root, eval_split, val_transform, keep_metadata)
    # 仅评估时不必构建训练集（避免多余的文件索引与预读开销）
    train_dst = None if opts.test_only else Odin1(opts.data_root, 'training', train_transform)
    return train_dst, val_dst


def build_model(opts, pretrained_backbone=False):
    """按参数构造模型；是否加载 ImageNet 主干由调用方决定。"""
    model = network.modeling.__dict__[opts.model](
        num_classes=Odin1.num_classes, output_stride=opts.output_stride,
        pretrained_backbone=pretrained_backbone)
    # 可选：替换为深度可分离卷积（只对 deeplabv3plus_* 生效）
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    # 把主干 BN 动量调小，让预训练统计量缓慢漂移
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    return model


def checkpoint_metadata(opts, config):
    """生成要写进检查点的语义元数据（用于加载时的一致性校验）。"""
    return {
        'dataset': 'odin1', 'num_classes': Odin1.num_classes,
        'class_names': list(Odin1.class_names),           # ("不可通行","可通行","天空")
        'raw_to_train_id': dict(Odin1.raw_to_train_id),   # {0:0, 255:1, 128:2}
        'model': opts.model, 'output_stride': opts.output_stride,
        'separable_conv': opts.separable_conv,
        'preprocess_config': config.to_dict(),            # 尺寸/归一化的完整快照
    }


def load_checkpoint(path, model, opts, config):
    """加载检查点：先校验语义元数据，再严格加载权重。"""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: %s' % path)
    # weights_only=True：只允许张量与基础类型，防止恶意 pickle
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint:
        raise ValueError('Expected a project checkpoint with model_state: %s' % path)
    expected = checkpoint_metadata(opts, config)
    # ---- 逐项比对"语义"字段：不一致直接报错 ----
    # 注意：只有"检查点里存在该键"时才比对；缺失的键留给下面的警告逻辑
    for key in ('dataset', 'num_classes', 'class_names', 'raw_to_train_id',
                'model', 'output_stride', 'separable_conv'):
        if key in checkpoint and checkpoint[key] != expected[key]:
            raise ValueError('Checkpoint %s mismatch: saved=%r, requested=%r' %
                             (key, checkpoint[key], expected[key]))
    # 类别语义缺失时无法自动校验，仅提示（例如旧格式检查点）
    if 'class_names' not in checkpoint or 'raw_to_train_id' not in checkpoint:
        print('Checkpoint lacks class metadata; class semantics cannot be verified automatically.')
    # ---- 预处理配置：只警告、不报错 ----
    # 原因：尺寸不一致仍可能被有意使用（如换分辨率评估），但必须让用户知情
    saved = checkpoint.get('preprocess_config')
    if saved is None:
        print('Checkpoint lacks preprocessing metadata; using the configuration declared in main.py.')
    elif saved != expected['preprocess_config']:
        print('Preprocessing differs from training. Saved: %s; current: %s' %
              (json.dumps(saved), json.dumps(expected['preprocess_config'])))
    # ---- 严格加载权重：键或形状不符即失败 ----
    try:
        model.load_state_dict(checkpoint['model_state'], strict=True)
    except RuntimeError as error:
        raise ValueError('Checkpoint does not match the selected three-class model: %s' % error) from error
    print('Loaded checkpoint: %s' % path)
    return checkpoint


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Metrics use network resolution; result PNGs use the retained crop resolution."""
    # 说明：labels/preds 都是网络分辨率（544×640），指标在此分辨率上统计；
    #      结果图则用 metadata 里保留的裁剪分辨率（1088×1280）绘制。
    metrics.reset()
    ret_samples = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=loader.dataset.split)):
            # 批次可能是 2 元组 (image,label) 或 3 元组 (image,label,metadata)
            images, labels = batch[:2]
            outputs = model(images.to(device, dtype=torch.float32))
            # 形状断言：通道数必须是 3，空间尺寸必须与标签一致
            if outputs.shape[1] != Odin1.num_classes or outputs.shape[-2:] != labels.shape[-2:]:
                raise ValueError('Model output and three-class label shape mismatch')
            # argmax 得到每像素类别索引（仍是网络分辨率）
            preds = outputs.argmax(1).cpu().numpy()
            # 累计混淆矩阵（在网络分辨率上）
            metrics.update(labels.numpy(), preds)
            # 是否需要出图：全量保存 或 该批次被选中用于可视化
            selected = ret_samples_ids is not None and batch_index in ret_samples_ids
            if opts.save_val_results or selected:
                # 出图必须有全尺寸元数据，因此 Dataset 必须开启 keep_metadata
                if len(batch) != 3:
                    raise ValueError('Full-size metadata is required for visualization')
                metadata = batch[2]
                for j, pred in enumerate(preds):
                    # make_visuals 内部会先把 pred 最近邻放大到 1088×1280，
                    # 再与裁剪后的原图按 0.6/0.4 混合生成 overlay
                    visuals = make_visuals(metadata['cropped_rgb'][j].numpy(), pred,
                                           loader.dataset.decode_target,
                                           metadata['cropped_target'][j].numpy())
                    sid = metadata['sample_id'][j]
                    if opts.save_val_results:
                        # 写入 results/<split>/<sample_id>_<类型>.png
                        save_sample_results(Path(opts.results_dir) / loader.dataset.split, sid, visuals)
                    # 只为每个批次取第一个样本做 visdom 展示，避免刷屏
                    if selected and j == 0:
                        ret_samples.append((sid, visuals))
    return metrics.get_results(), ret_samples


def report_score(score, config):
    """打印指标，并显式标注"指标是在哪个分辨率上算的"。"""
    h, w = config.input_size_hw
    print('Metric resolution WH: %dx%d' % (w, h))    # 例如 640x544（宽×高）
    print(StreamSegMetrics.to_str(score))            # Overall Acc / Mean Acc / Mean IoU 等
    for class_id, name in enumerate(Odin1.class_names):
        print('Class %d (%s) IoU: %.6f' % (class_id, name, score['Class IoU'][class_id]))


def main():
    parser = get_argparser()
    opts = parser.parse_args()
    # ---- 1. 参数合法性校验（全部通过 parser.error 报错，退出码为 2）----
    for key in ('batch_size', 'val_batch_size', 'total_itrs', 'step_size', 'print_interval', 'val_interval'):
        if getattr(opts, key) <= 0:
            parser.error('%s must be positive' % key)
    if opts.num_workers < 0 or opts.vis_num_samples < 0:
        parser.error('num_workers and vis_num_samples must be non-negative')
    # 评估/续训必须有检查点
    if (opts.test_only or opts.continue_training) and not opts.ckpt:
        parser.error('--test_only and --continue_training require --ckpt')
    if opts.ckpt and not Path(opts.ckpt).is_file():
        parser.error('Checkpoint not found: %s' % opts.ckpt)
    # ASPP 中的 BatchNorm 在单样本 batch 下统计量无效
    if not opts.test_only and opts.batch_size < 2:
        parser.error('Training batch_size must be at least 2 for the existing ASPP BatchNorm')

    # ---- 2. 设备与随机性 ----
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # ---- 3. 预处理配置（唯一真源在这里被实例化）----
    config = get_preprocess_config()
    print('Device: %s; preprocessing: %s' % (device, json.dumps(config.to_dict())))

    # ---- 4. visdom 可选（惰性导入，未安装也不影响训练）----
    vis = None
    if opts.enable_vis:
        from utils.visualizer import Visualizer
        vis = Visualizer(port=opts.vis_port, env=opts.vis_env)
        vis.vis_table('Options', vars(opts))

    # ---- 5. 数据集与验证 DataLoader ----
    train_dst, val_dst = get_dataset(opts, config)
    # 验证集顺序遍历：保证结果文件编号稳定、可复现
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=False,
                                 num_workers=opts.num_workers)
    print('odin1 %s: %d samples' % (val_dst.split, len(val_dst)))

    # ---- 6. 建模型 ----
    # 只有"非评估 且 无检查点 且 未显式禁用"时才加载 ImageNet 主干预训练权重
    model = build_model(opts, pretrained_backbone=(
        not opts.test_only and not opts.ckpt and not opts.no_pretrained_backbone))
    # 有检查点则加载（先 load_state_dict，后 to(device)）
    checkpoint = load_checkpoint(opts.ckpt, model, opts, config) if opts.ckpt else None
    model.to(device)
    # 仅当有多张 GPU 时才包 DataParallel（单卡保持原始模型，便于取 backbone/classifier）
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # 三分类专用指标：mIoU / Acc / 各类 IoU / 可通行 P-R
    metrics = StreamSegMetrics(Odin1.num_classes)
    # visdom 采样批次索引（注意是 batch 索引，不是样本索引）
    sample_ids = set(range(min(opts.vis_num_samples, len(val_loader)))) if vis else None

    def evaluate():
        """跑一遍验证集、打印指标、按需推送 visdom；返回指标字典。"""
        score, samples = validate(opts, model, val_loader, device, metrics, sample_ids)
        report_score(score, config)
        if vis:
            for sid, visuals in samples:
                for name in ('image', 'target_color', 'overlay'):
                    # PIL HWC → CHW，visdom 需要通道在前
                    vis.vis_image('%s %s' % (sid, name), visuals[name].transpose(2, 0, 1))
        return score

    # ---- 7. 仅评估模式：跑一轮就退出 ----
    if opts.test_only:
        evaluate()
        return

    # ---- 8. 训练 DataLoader ----
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,
                                   num_workers=opts.num_workers, drop_last=True)
    # drop_last=True 后若一个批次都没有，说明 batch_size 大于训练样本数，直接报错
    if len(train_loader) == 0:
        parser.error('batch_size exceeds training sample count; reduce --batch_size')

    # ---- 9. 优化器：双学习率（主干 0.1×，分类头 1×）----
    # 这里显式处理 DataParallel：包装后参数挂在 model.module 上
    optimizer = torch.optim.SGD([
        {'params': model.module.backbone.parameters() if isinstance(model, nn.DataParallel)
         else model.backbone.parameters(), 'lr': opts.lr * 0.1},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel)
         else model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # ---- 10. 学习率调度器 ----
    # poly：lr × (1 - it/max_iters)^0.9，在训练结束时衰减到下限
    # step：每 step_size 次更新把 lr 乘以 0.1
    scheduler = (utils.PolyLR(optimizer, opts.total_itrs, power=0.9) if opts.lr_policy == 'poly'
                 else torch.optim.lr_scheduler.StepLR(optimizer, opts.step_size, gamma=0.1))

    # ---- 11. 损失函数 ----
    # focal_loss 的 FocalLoss 默认 gamma=0，实际近似退化为交叉熵；ignore_index=255 兼容旧标签约定
    criterion = (utils.FocalLoss(ignore_index=255, size_average=True) if opts.loss_type == 'focal_loss'
                 else nn.CrossEntropyLoss(ignore_index=255))

    # ---- 12. 训练状态初始化 ----
    cur_itrs, cur_epochs, best_score = 0, 0, 0.0
    if opts.continue_training:
        # 严格续训：四个字段缺一不可
        for key in ('optimizer_state', 'scheduler_state', 'cur_itrs', 'best_score'):
            if key not in checkpoint:
                raise ValueError('Checkpoint cannot resume training: missing %s' % key)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        cur_itrs, best_score = int(checkpoint['cur_itrs']), float(checkpoint['best_score'])
        # cur_epochs 是可选字段（旧检查点可能没有），仅用于日志显示
        cur_epochs = int(checkpoint.get('cur_epochs', 0))
    del checkpoint   # 权重已装进模型，释放检查点占用的内存

    # ---- 13. 检查点目录与保存闭包 ----
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    def save_ckpt(kind):
        """保存检查点；kind 取 'best' 或 'latest'。"""
        # DataParallel 包装时取内部的原始模型，保证 state_dict 没有 "module." 前缀
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        # 元数据（类别语义 + 预处理快照）
        payload = checkpoint_metadata(opts, config)
        # 追加训练状态
        payload.update(cur_itrs=int(cur_itrs), cur_epochs=int(cur_epochs), best_score=float(best_score),
                       model_state=raw_model.state_dict(), optimizer_state=optimizer.state_dict(),
                       scheduler_state=scheduler.state_dict())
        # 命名规则：<best|latest>_<模型名>_odin1_os<stride>.pth
        path = checkpoint_dir / ('%s_%s_odin1_os%d.pth' % (kind, opts.model, opts.output_stride))
        torch.save(payload, path)
        print('Model saved: %s' % path)

    # ---- 14. 训练主循环（按 iteration 驱动）----
    interval_loss, interval_count = 0.0, 0    # 用于计算区间平均 loss
    while cur_itrs < opts.total_itrs:
        model.train()
        cur_epochs += 1
        for images, labels in train_loader:
            # 标准训练四步
            optimizer.zero_grad()
            loss = criterion(model(images.to(device, dtype=torch.float32)),
                             labels.to(device, dtype=torch.long))
            loss.backward()
            optimizer.step()
            scheduler.step()          # 按 iteration 步进（PolyLR 的步长是 iteration）
            cur_itrs += 1
            interval_loss += loss.item()
            interval_count += 1
            if vis:
                vis.vis_scalar('Loss', cur_itrs, loss.item())
            # 每 print_interval 次更新打印一次区间平均 loss
            if cur_itrs % opts.print_interval == 0:
                print('Epoch %d, iteration %d/%d, loss %.6f' %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss / interval_count))
                interval_loss, interval_count = 0.0, 0
            # 到达验证周期：验证 → 存 best（若创新高）→ 存 latest
            if cur_itrs % opts.val_interval == 0:
                score = evaluate()
                if score['Mean IoU'] > best_score:
                    best_score = float(score['Mean IoU'])
                    save_ckpt('best')
                save_ckpt('latest')
                if vis:
                    vis.vis_scalar('[Val] Mean IoU', cur_itrs, score['Mean IoU'])
                    vis.vis_table('[Val] Class IoU', score['Class IoU'])
                model.train()          # 验证会切到 eval，这里切回训练模式
            # 精确停在 total_itrs：跳出内层循环，外层 while 条件随之结束
            if cur_itrs >= opts.total_itrs:
                break
    # 循环结束后再存一次 latest，确保最终状态落盘
    save_ckpt('latest')


if __name__ == '__&#8203;main__':
    main()
