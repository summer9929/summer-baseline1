# -*- coding: utf-8 -*-
"""Quick visual montages for the segdataodin audit (read-only on the dataset)."""
import os
from PIL import Image
import numpy as np

ROOT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\segdataodin"
OUT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\wkb\audit_20261124"

CMAP = np.array([(220, 50, 50), (60, 140, 240), (40, 200, 60)], np.uint8)  # 0 imp, 128 sky, 255 pass
RAW2ID = {0: 0, 128: 1, 255: 2}


def load(split, sid):
    rgb = Image.open(os.path.join(ROOT, split, 'image_data', sid + '.png')).convert('RGB')
    gt = np.array(Image.open(os.path.join(ROOT, split, 'gt_image', sid + '.png')))
    dep = np.array(Image.open(os.path.join(ROOT, split, 'dense_depth', sid + '.png'))).astype(np.float32) / 256.0
    return np.array(rgb), gt, dep


def colorize_gt(gt):
    ids = np.zeros_like(gt, dtype=np.uint8)
    for k, v in RAW2ID.items():
        ids[gt == k] = v
    return CMAP[ids]


def colorize_depth(dep, vmax=None):
    vmax = vmax or float(np.percentile(dep[dep > 0], 99)) if dep.max() > 0 else 1.0
    x = np.clip(dep / max(vmax, 1e-3), 0, 1)
    # turbo-ish: simple 3-stop ramp
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    img = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    img[dep <= 0] = 0
    return img


def shrink(a, f=3):
    return np.array(Image.fromarray(a).resize((a.shape[1] // f, a.shape[0] // f), Image.BILINEAR))


def band(items, gap=6):
    hs = [i.shape[0] for i in items]
    H = max(hs)
    tot = sum(i.shape[1] for i in items) + gap * (len(items) - 1)
    out = np.full((H, tot, 3), 255, np.uint8)
    x = 0
    for i in items:
        out[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


def stack(rows, gap=6):
    w = max(r.shape[1] for r in rows)
    tot = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.full((tot, w, 3), 255, np.uint8)
    y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def edge_overlay(rgb, dep, thr=0.15):
    """RGB with depth-invalid boundary in cyan and depth jump in yellow."""
    g = rgb.astype(np.float32).mean(2)
    img = rgb.copy()
    inv = (dep <= 0)
    b = np.zeros_like(inv)
    b[:, 1:] |= inv[:, 1:] != inv[:, :-1]
    b[1:, :] |= inv[1:, :] != inv[:-1, :]
    img[b] = (0, 255, 255)
    # depth jump
    j = np.zeros_like(dep)
    j[:, 1:] = np.abs(np.diff(dep, axis=1))
    j[1:, :] = np.maximum(j[1:, :], np.abs(np.diff(dep, axis=0)))
    jm = (j > thr) & (dep > 0)
    img[jm] = (255, 255, 0)
    return img


def main():
    samples = [('training', '346035329872'), ('training', '681150656293'), ('validation', '817224311091')]
    rows = []
    for split, sid in samples:
        rgb, gt, dep = load(split, sid)
        r1 = band([shrink(rgb), shrink(colorize_gt(gt)), shrink(colorize_depth(dep)),
                   shrink(edge_overlay(rgb, dep))])
        rows.append(r1)
    Image.fromarray(stack(rows)).save(os.path.join(OUT, 'fig01_montage.png'))
    print('wrote fig01_montage.png')

    # crop comparison for one frame
    rgb, gt, dep = load('training', '346035329872')
    box = (160, 104, 1440, 1192)
    c_rgb = rgb[box[1]:box[3], box[0]:box[2]]
    c_gt = gt[box[1]:box[3], box[0]:box[2]]
    c_dep = dep[box[1]:box[3], box[0]:box[2]]
    Image.fromarray(stack([band([shrink(c_rgb), shrink(colorize_gt(c_gt)), shrink(colorize_depth(c_dep)),
                                 shrink(edge_overlay(c_rgb, c_dep))])])).save(
        os.path.join(OUT, 'fig02_crop.png'))
    print('wrote fig02_crop.png', c_rgb.shape)


if __name__ == '__main__':
    main()
