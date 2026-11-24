# -*- coding: utf-8 -*-
"""1:1 zoom panels: RGB vs GT boundary vs depth boundary, to judge registration/label accuracy."""
import os
from PIL import Image
import numpy as np

ROOT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\segdataodin"
OUT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\wkb\audit_20261124"
CMAP = np.array([(220, 50, 50), (60, 140, 240), (40, 200, 60)], np.uint8)
RAW2ID = {0: 0, 128: 1, 255: 2}


def load(split, sid):
    rgb = np.array(Image.open(os.path.join(ROOT, split, 'image_data', sid + '.png')).convert('RGB'))
    gt = np.array(Image.open(os.path.join(ROOT, split, 'gt_image', sid + '.png')))
    dep = np.array(Image.open(os.path.join(ROOT, split, 'dense_depth', sid + '.png'))).astype(np.float32) / 256.0
    return rgb, gt, dep


def gt_color(gt):
    ids = np.zeros_like(gt, dtype=np.uint8)
    for k, v in RAW2ID.items():
        ids[gt == k] = v
    return CMAP[ids]


def gt_boundary(gt):
    b = np.zeros(gt.shape, bool)
    b[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    b[1:, :] |= gt[1:, :] != gt[:-1, :]
    return b


def band(items, gap=8):
    H = max(i.shape[0] for i in items)
    W = sum(i.shape[1] for i in items) + gap * (len(items) - 1)
    o = np.full((H, W, 3), 255, np.uint8)
    x = 0
    for i in items:
        o[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return o


def stack(rows, gap=8):
    W = max(r.shape[1] for r in rows)
    H = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    o = np.full((H, W, 3), 255, np.uint8)
    y = 0
    for r in rows:
        o[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    return o


def main():
    # region of interest: left tent structure (strong straight edges) + near road boundary
    rgb, gt, dep = load('training', '346035329872')
    b = gt_boundary(gt)
    ov = rgb.copy()
    ov[b] = (255, 0, 255)          # magenta = GT boundary
    boxes = [(0, 0, 420, 380), (520, 600, 940, 980), (1050, 620, 1470, 1000)]
    rows = []
    for (x0, y0, x1, y1) in boxes:
        rows.append(band([rgb[y0:y1, x0:x1], ov[y0:y1, x0:x1], gt_color(gt)[y0:y1, x0:x1]]))
    Image.fromarray(stack(rows)).save(os.path.join(OUT, 'fig03_zoom_gt_boundary.png'))
    print('fig03 ok')

    # tractor close-up (validation frame) - object silhouette is the registration stress test
    rgb2, gt2, dep2 = load('validation', '817224311091')
    b2 = gt_boundary(gt2)
    ov2 = rgb2.copy()
    ov2[b2] = (255, 0, 255)
    x0, y0, x1, y1 = 700, 500, 1250, 1050
    Image.fromarray(band([rgb2[y0:y1, x0:x1], ov2[y0:y1, x0:x1], gt_color(gt2)[y0:y1, x0:x1]])).save(
        os.path.join(OUT, 'fig04_zoom_tractor.png'))
    print('fig04 ok')

    # depth block structure zoom
    a = np.array(Image.open(os.path.join(ROOT, 'training', 'dense_depth', '346035329872.png')))
    patch = a[1150:1230, 690:790]
    img = np.zeros(patch.shape + (3,), np.uint8)
    v = np.clip((patch.astype(np.float32) / max(patch.max(), 1)) * 255, 0, 255).astype(np.uint8)
    img[..., 0] = v; img[..., 1] = v; img[..., 2] = v
    img[patch == 0] = (40, 40, 40)
    Image.fromarray(img).resize((800, 640), Image.NEAREST).save(os.path.join(OUT, 'fig05_depth_blocks.png'))
    print('fig05 ok', patch.shape)


if __name__ == '__main__':
    main()
