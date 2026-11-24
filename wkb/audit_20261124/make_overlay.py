# -*- coding: utf-8 -*-
"""深度-图像叠加图：直接目视判断深度是否落在正确物体上（1:1 局部放大）。"""
import os
from PIL import Image
import numpy as np

ROOT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\segdataodin"
OUT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\wkb\audit_20261124"


def load(sp, sid):
    rgb = np.array(Image.open(os.path.join(ROOT, sp, 'image_data', sid + '.png')).convert('RGB'))
    dep = np.array(Image.open(os.path.join(ROOT, sp, 'dense_depth', sid + '.png'))).astype(np.float32) / 256.0
    return rgb, dep


def depth_rgb(dep, vmax=12.0):
    x = np.clip(dep / vmax, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    o = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    o[dep <= 0] = 255
    return o


def overlay(rgb, dep, alpha=0.55):
    d = depth_rgb(dep)
    m = (dep <= 0)[..., None]
    o = (rgb.astype(np.float32) * (1 - alpha) + d.astype(np.float32) * alpha)
    o = np.where(m, rgb.astype(np.float32) + 0.45 * (255 - rgb.astype(np.float32)), o)
    return np.clip(o, 0, 255).astype(np.uint8)


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
    cases = [('validation', '817224311091', (700, 520, 1220, 1040), 'tractor_near'),
             ('training', '681150656293', (700, 560, 1220, 1080), 'tractor_far'),
             ('training', '346035329872', (60, 40, 580, 560), 'tent')]
    rows = []
    for sp, sid, (x0, y0, x1, y1), nm in cases:
        rgb, dep = load(sp, sid)
        a = rgb[y0:y1, x0:x1]
        b = overlay(rgb, dep)[y0:y1, x0:x1]
        c = depth_rgb(dep)[y0:y1, x0:x1]
        rows.append(band([a, b, c]))
        Image.fromarray(band([a, b, c])).save(os.path.join(OUT, 'fig06_overlay_%s.png' % nm))
    Image.fromarray(stack(rows)).save(os.path.join(OUT, 'fig06_overlay_all.png'))
    print('ok')


if __name__ == '__main__':
    main()
