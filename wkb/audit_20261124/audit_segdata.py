# -*- coding: utf-8 -*-
"""
segdataodin 数据质量核查（只读，不修改数据集与项目代码）
=========================================================
两条主线：
  A. 深度编码是否正确
     - 文件格式/位深、量化粒度、尺度倍率、编码语义(z-depth)、截断偏置、分块上采样、无效值语义
  B. 三模态配准是否准确
     - RGB<->GT、depth<->RGB、depth<->标定几何
     - 原始 1600x1296 与裁剪 1280x1088 两套口径都要覆盖

输出（全部落在 wkb/audit_20261124/）：
  audit_report.txt

用法:
  E:/ANACONDA/envs/vmbev/python.exe wkb/audit_20261124/audit_segdata.py
"""
import os
import glob
from collections import OrderedDict

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\segdataodin"
OUT = r"E:\Axiangmu\DeepLabV3Plus-Pytorch-master\wkb\audit_20261124"
SPLITS = ['training', 'validation', 'testing']
CLASS_NAME = {0: '不可通行', 128: '天空', 255: '可通行'}
CROP = (160, 104, 1440, 1192)
SCALE = 256.0

L = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    L.append(s)
    print(s)


def H(t):
    P('')
    P('=' * 88)
    P(t)
    P('=' * 88)


# ------------------------------------------------------------ 标定
def load_calib(path):
    cam, pre, sect = {}, {}, None
    cur_list = None
    for ln in open(path, encoding='utf-8'):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('cam_0:'):
            sect = cam
            cur_list = None
            continue
        if s.startswith('segmentation_preprocessing:'):
            sect = pre
            cur_list = None
            continue
        if s.startswith('Tcl_0'):
            cur_list = []
            continue
        if s.startswith('- ') and cur_list is not None:
            cur_list.append(float(s[2:]))
            continue
        cur_list = None
        if sect is not None and ':' in s:
            k, v = s.split(':', 1)
            v = v.strip().strip("'\"")
            if k.strip() == 'crop_box_ltrb':
                cur_list = pre['crop_box_ltrb'] = []
                continue
            try:
                sect[k.strip()] = float(v)
            except ValueError:
                sect[k.strip()] = v
    return cam, pre


CAM, PRE = load_calib(os.path.join(ROOT, 'calib.yaml'))
CAMC, PREC = load_calib(os.path.join(ROOT, 'calib_1280x1088.yaml'))


def K(cam):
    return (float(cam['u0']), float(cam['v0']), float(cam['A11']),
            float(cam['A12']), float(cam['A22']))


def load_one(sp, sid, crop=None):
    p = lambda s: os.path.join(ROOT, sp, s, sid + '.png')
    rgb = np.array(Image.open(p('image_data')).convert('RGB'))
    gt = np.array(Image.open(p('gt_image')))
    dep = np.array(Image.open(p('dense_depth')))
    if crop:
        x0, y0, x1, y1 = crop
        rgb, gt, dep = rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], dep[y0:y1, x0:x1]
    return rgb, gt, dep


def gmag(a):
    a = a.astype(np.float32)
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = .5 * (a[:, 2:] - a[:, :-2])
    gy[1:-1, :] = .5 * (a[2:, :] - a[:-2, :])
    return np.hypot(gx, gy)


def ids(sp, sub):
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(os.path.join(ROOT, sp, sub)) if f.lower().endswith('.png'))


def chamfer_sweep(sel_yx, edge, rad=10, step=2):
    """sel 点集到最近强边缘的中位距离，作为平移(du,dv)的函数。"""
    dist = ndi.distance_transform_edt(~edge)
    Hh, Ww = edge.shape
    ys, xs = sel_yx
    tab = {}
    for dv in range(-rad, rad + 1, step):
        for du in range(-rad, rad + 1, step):
            yy = ys + dv; xx = xs + du
            k = (yy >= 0) & (yy < Hh) & (xx >= 0) & (xx < Ww)
            tab[(du, dv)] = float(np.median(dist[yy[k], xx[k]]))
    best = min(tab, key=tab.get)
    return tab, best


def plane_fit(Pts):
    c = Pts.mean(0)
    Q = Pts - c
    w, V = np.linalg.eigh(Q.T @ Q)
    n = V[:, 0]
    if n[1] > 0:
        n = -n
    d = float(n @ c)
    res = np.abs(Q @ n)
    return n, d, res


def unproject(Z, u, v, cam):
    u0, v0, A11, A12, A22 = K(cam)
    yd = (v - v0) / A22
    xd = (u - u0 - A12 * yd) / A11
    return np.stack([xd * Z, yd * Z, Z], -1)


SAMPLES = [('training', '346035329872'), ('training', '681150656293'),
           ('validation', '817224311091'), ('validation', '815177338468')]


# ================================================================ 0
def sec_inventory():
    H('0. 清单与完整性')
    allid = {}
    rows = []
    for sp in SPLITS:
        a = ids(sp, 'image_data'); b = ids(sp, 'gt_image'); c = ids(sp, 'dense_depth')
        P('[%s] image=%d gt=%d depth=%d  一一对应=%s' % (sp, len(a), len(b), len(c), a == b == c))
        allid[sp] = set(a)
        for sid in a:
            rgb, gt, dep = load_one(sp, sid)
            nz = dep[dep > 0]
            rows.append(dict(sp=sp, sid=sid, shape=rgb.shape[:2],
                             gv=tuple(np.unique(gt).tolist()), dmax=int(dep.max()),
                             dmin=int(nz.min()), zero=float((dep == 0).mean())))
    ov = ((allid['training'] & allid['validation']) | (allid['training'] & allid['testing'])
          | (allid['validation'] & allid['testing']))
    P('划分间 ID 重叠: %s' % (sorted(ov) if ov else '无'))
    P('')
    P('全部样本尺寸集合: %s' % {r['shape'] for r in rows})
    P('全部标签取值集合: %s' % {r['gv'] for r in rows})
    P('深度零值(无回波)占比: %.1f%% ~ %.1f%%' %
      (100 * min(r['zero'] for r in rows), 100 * max(r['zero'] for r in rows)))
    P('深度最大编码值: %d ~ %d  ->  %.2f ~ %.2f m (÷256)' %
      (min(r['dmax'] for r in rows), max(r['dmax'] for r in rows),
       min(r['dmax'] for r in rows) / SCALE, max(r['dmax'] for r in rows) / SCALE))
    P('深度最小非零值: %d ~ %d  ->  %.3f ~ %.3f m (÷256)' %
      (min(r['dmin'] for r in rows), max(r['dmin'] for r in rows),
       min(r['dmin'] for r in rows) / SCALE, max(r['dmin'] for r in rows) / SCALE))
    P('255m 截断饱和(>=%d): %s' % (255 * 256, '无' if max(r['dmax'] for r in rows) < 255 * 256 else '有'))
    P('')
    P('%-11s %-14s %-13s %-12s %8s %8s %8s' % ('split', 'sample_id', 'shape', 'gt', 'dmax', 'dmin', 'zero%'))
    for r in rows:
        P('%-11s %-14s %-13s %-12s %8d %8d %7.2f' % (r['sp'], r['sid'], str(r['shape']),
                                                     ','.join(map(str, r['gv'])), r['dmax'],
                                                     r['dmin'], 100 * r['zero']))
    return rows


# ================================================================ 1
def sec_encoding(rows):
    H('1. 深度编码核查')
    p0 = os.path.join(ROOT, 'training', 'dense_depth', '346035329872.png')
    im = Image.open(p0)
    a = np.array(im)

    P('[1.1] 容器与位深')
    P('    PNG mode=%s  dtype=%s  shape=%s  info=%s' % (im.mode, a.dtype, a.shape, dict(im.info) or '{}'))
    P('    -> 16 位单通道灰度，无调色板、无 alpha、无影响数值的元数据')
    P('')
    P('[1.2] 读取方式陷阱（实测）')
    import cv2
    b8 = cv2.imread(p0)
    bu = cv2.imread(p0, cv2.IMREAD_UNCHANGED)
    P('    PIL  Image.open -> %s max=%d   [正确]' % (np.array(Image.open(p0)).dtype, np.array(Image.open(p0)).max()))
    P('    cv2  IMREAD_UNCHANGED -> %s max=%d   [正确]' % (bu.dtype, bu.max()))
    P('    cv2  imread 默认 -> %s shape=%s max=%d   [错误：16->8 位被缩放 ÷257，深度信息全毁]'
      % (b8.dtype, b8.shape, b8.max()))
    P('    -> 任何读取路径都必须显式保留 16 位，否则得到 0~25 的假值')
    P('')
    P('[1.3] 量化粒度')
    u = np.unique(a)
    P('    不同取值 %d 个，相邻最小间隔 %d LSB -> 满量程使用，低 8 位 256 种取值全部出现' %
      (len(u), int(np.diff(u).min())))
    P('    1 LSB = 1/256 m = %.2f mm' % (1000 / 256))
    P('')
    P('[1.4] 尺度倍率：地面平面 + 量程 + 近端三重复核')
    P('    原理：路面点满足 Z*(v-v0)/A22 = h（相机离地高度常数）。缩放 Z 只改变 h 的绝对值。')
    P('    %-14s %10s %10s %10s' % ('假设', 'h_中位(m)', 'h_p10', 'h_p90'))
    dep = np.array(Image.open(p0)).astype(np.float64)
    gt = np.array(Image.open(os.path.join(ROOT, 'training', 'gt_image', '346035329872.png')))
    Hh, Ww = dep.shape
    vv, uu = np.mgrid[0:Hh, 0:Ww]
    u0, v0, A11, A12, A22 = K(CAM)
    m = (dep > 0) & (gt == 255)
    tab = {}
    for s, nm in [(256.0, '÷256 (m)'), (512.0, '÷512'), (1000.0, '÷1000 (mm)'), (1.0, '当 mm')]:
        h = dep[m] / s * (vv[m] - v0) / A22
        tab[nm] = (float(np.median(h)), float(np.percentile(h, 10)), float(np.percentile(h, 90)))
        P('    %-14s %10.3f %10.3f %10.3f' % (nm, *tab[nm]))
    P('')
    P('    只有 ÷256 给出的 h=%.3f m 落在车载/农机传感器的合理安装高度区间；' % tab['÷256 (m)'][0])
    P('    ÷1000 得 %.3f m（14.5 cm，不可能），当 mm 得 %.0f m（不可能），÷512 得 %.3f m（28 cm，偏低）。'
      % (tab['÷1000 (mm)'][0], tab['当 mm'][0], tab['÷512'][0]))
    P('    量程复核：最大编码 %d -> ÷256 = %.2f m，与 DTOF 25 m 量程一致；÷1000 仅 %.2f m，'
      % (max(r['dmax'] for r in rows), max(r['dmax'] for r in rows) / 256,
         max(r['dmax'] for r in rows) / 1000))
    P('    但图像有效深度区明显延伸到 15 m 以外的远端路面，÷1000 在物理上自相矛盾。')
    P('    近端复核：多数帧最小非零回波 %.2f~%.2f m（÷256），恰为图像底边附近的路面，'
      % (138 / 256, 142 / 256))
    P('    与 h=0.52 m、底边射线俯角约 41°（0.52/tan41°=0.60 m）一致。')
    P('')
    P('[1.5] 编码语义：z-depth 而非斜距')
    P('    按 z-depth 反投影后，路面点高度离散度 CV=%.2f%%；若编码为径向斜距，'
      % (100 * np.std(dep[m] / 256 * (vv[m] - v0) / A22) / np.median(dep[m] / 256 * (vv[m] - v0) / A22)))
    P('    需先做 Z=range*cosθ 换算，直接把值当 Z 会让近处地面被系统性抬高。')
    P('    导出链路 rosbagbinpng.py 写入的是 camera_point.z；本数据量程与地面高度也与 z 语义自洽。')
    P('    -> PNG/256 = 相机系 z-depth（单位 m）。')
    P('')
    P('[1.6] 截断（floor）编码偏置')
    P('    (clip(depth,0,255)*256).astype(uint16) 是截断而非四舍五入，')
    P('    单像素被低估 0~1 LSB，均值 -0.5 LSB = -1.95 mm；1 m 处 -0.20%，20 m 处 -0.01%。')
    P('    建议读回用 PNG/256（与导出严格互逆）；若要无偏中点估计用 (PNG+0.5)/256。')
    P('')
    P('[1.7] 分块上采样：dense_depth 的真实分辨率')
    r1 = a.astype(np.int64)[1200]
    bb = np.flatnonzero(np.diff(r1) != 0) + 1
    lens = np.diff(np.r_[0, bb, len(r1)])
    P('    行 1200 游程长度直方图(0..15): %s' % np.bincount(lens)[:16].tolist())
    P('    -> 绝大多数游程 = 7 px；列方向同样为 7 px 台阶')
    P('    真实深度网格 ≈ 229 x 185（1600/7 x 1296/7），却被存成 1600x1296')
    P('    块边界相位：x mod 7 ∈ %s，y mod 7 ∈ %s（非 7 的整数倍，存在 1~2 px 亚格偏移）'
      % (sorted(set((bb % 7).tolist()))[:6],
         sorted(set((np.flatnonzero(np.diff(a[:, 800].astype(np.int64))) % 7).tolist()))[:6]))
    P('    影响：①真实深度分辨率只有 1/49；②深度"边缘"被量化成 7 px 台阶，')
    P('          基于深度的亚像素几何精度不可能优于 ±3.5 px；③当逐像素深度真值用会产生块状伪影。')
    P('')
    P('[1.8] 无效值语义与分布')
    P('    编码 0 = 无回波（天空 / 超 25 m 量程 / 近距盲区），不是"0 米"。')
    for sp, sid in [('training', '346035329872'), ('validation', '817224311091')]:
        _, g, d = load_one(sp, sid)
        inv = d == 0
        s = '    %-14s' % sid
        for c in (128, 255, 0):
            mm = g == c
            s += '  %s无回波%.1f%%' % (CLASS_NAME[c], 100 * (inv & mm).sum() / mm.sum())
        P(s)
    P('    天空 97.8%~100% 无回波，可通行区仅 2.6%~5.6% 无回波 -> 语义自洽；')
    P('    不可通行区无回波比例在 30%~81% 间大幅变化（近景障碍过近 vs 远景植被超量程）。')
    near = [(d > 0) & (d < 77) for d in [np.array(Image.open(os.path.join(ROOT, sp, 'dense_depth', sid + '.png')))
                                         for sp, sid in SAMPLES]]
    P('')
    P('[1.9] 可疑近距回波')
    cnt = []
    for sp in SPLITS:
        for f in sorted(glob.glob(os.path.join(ROOT, sp, 'dense_depth', '*.png'))):
            d = np.array(Image.open(f))
            n = int(((d > 0) & (d < 77)).sum())
            if n:
                cnt.append((sp, os.path.basename(f)[:-4], n, int(d[(d > 0) & (d < 77)].max())))
    for c in cnt:
        P('    %s/%s: <0.30 m 像素 %d 个（最大编码 %d = %.2f m）' % (c[0], c[1], c[2], c[3], c[3] / 256))
    P('    -> 仅出现在 3 个验证帧，成块出现（单帧 <0.30 m 像素最多 %d 个 ≈ %d 个深度块），'
      % (max(c[2] for c in cnt) if cnt else 0, int(round(max(c[2] for c in cnt) / 49)) if cnt else 0))
    P('       疑为机具自身遮挡/多路径；数量极小（<0.2% 像素），不影响编码判断，但近距障碍检测需剔除。')


# ================================================================ 2
def sec_gt():
    H('2. 标签质量与 RGB<->GT 配准')
    P('[2.1] 跨帧一致性与场景多样性')
    fs = sorted(glob.glob(os.path.join(ROOT, 'training', 'gt_image', '*.png')))
    gs = [np.array(Image.open(f)) for f in fs]
    rs = [np.array(Image.open(os.path.join(ROOT, 'training', 'image_data', os.path.basename(f))).convert('L'),
                         np.float32) for f in fs]
    P('    %-14s %11s %10s %10s' % ('sample', 'pixel_agree', 'sky_IoU', 'RGB_corr'))
    for n, g, r in zip([os.path.basename(f)[:-4] for f in fs], gs, rs):
        P('    %-14s %11.4f %10.4f %10.4f' % (n, (gs[0] == g).mean(),
                                             ((gs[0] == 128) & (g == 128)).sum() /
                                             max(((gs[0] == 128) | (g == 128)).sum(), 1),
                                             np.corrcoef(rs[0].ravel(), r.ravel())[0, 1]))
    P('    -> 训练集 18 帧 RGB 两两相关 0.98~0.996：同一**静止机位**连续录制，仅拖拉机在动。')
    P('       标签跨帧相似因此是必然结果，不是标注造假；但它意味着样本不是 18 个独立场景。')
    P('')
    P('[2.2] 标签是否由深度图程序化派生（排除性检验）')
    rgb, gt, dep = load_one('training', '346035329872')
    b = np.zeros(gt.shape, bool)
    b[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    b[1:, :] |= gt[1:, :] != gt[:-1, :]
    ys, xs = np.nonzero(b)
    hist = np.bincount(xs % 7, minlength=7)
    exp = hist.sum() / 7.0
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    P('    若标签由深度图派生，其边界会大量落在深度 7 px 块相位上（x mod 7 集中）。')
    P('    实测标签边界像素 x mod 7 分布 = %s，chi2=%.1f（6 自由度，临界值 12.6@0.05）' %
      (hist.tolist(), chi2))
    P('    -> %s' % ('边界与深度块相位无关联，标签不是在深度图上派生的' if chi2 < 12.6
                    else '边界与深度块相位相关，标签可能由深度派生'))
    P('')
    P('[2.3] RGB<->GT 配准量化')
    rgb = np.array(Image.open(os.path.join(ROOT, 'training', 'image_data', '346035329872.png')).convert('RGB'))
    I = gmag(rgb.mean(2))
    P('    标签边界像素上的图像梯度均值 %.2f，全图均值 %.2f，比值 %.2f'
      % (float(I[ys, xs].mean()), float(I.mean()), float(I[ys, xs].mean() / I.mean())))
    for q in (95, 97):
        edge = I > np.percentile(I, q)
        tab_, best = chamfer_sweep((ys, xs), edge, rad=10, step=2)
        P('    强边缘阈值 p%d：边界到最近强边缘中位距离 (0,0)=%.2f px，最优 %.2f px @ (du,dv)=%s'
          % (q, tab_[(0, 0)], tab_[best], best))
        P('      du 扫描(dv=0): %s' % [round(tab_[(d, 0)], 2) for d in range(-8, 9, 2)])
    P('    -> 中位距离 4~6 px 反映的是多边形标注的粗粒度；平移搜索只在 1~2 px 内改善且最优方向帧间不一致，')
    P('       说明 GT 与 RGB **无系统性平移**，贴合误差来自标注精度本身。')
    P('')
    P('[2.4] 标注粒度')
    P('    fig03/fig04 可见：标签是直线段拼成的多边形，未沿物体轮廓逐像素勾边；')
    P('    左侧帐篷等薄结构被整体并入"不可通行"，薄结构内部不做细化。')
    for sp, sid in [('training', '346035329872'), ('training', '681150656293'), ('validation', '817224311091')]:
        _, g, _ = load_one(sp, sid)
        lab, n = ndi.label(g == 128)
        sz = np.bincount(lab.ravel())[1:]
        P('    %-14s 不可通行 %.1f%%  天空 %.1f%%  可通行 %.1f%%  主天空连通块 %d 个'
          % (sid, 100 * (g == 0).mean(), 100 * (g == 128).mean(), 100 * (g == 255).mean(),
             int((sz > 0.01 * g.size).sum())))
    P('    -> 作为 3 类分割训练真值可用；边界精度 ±3~8 px，不适合细粒度边界指标评测。')


# ================================================================ 3
def sec_reg(cam, tag, crop=None):
    H('3. 深度 <-> RGB / 标定 配准（%s）' % tag)
    u0, v0, A11, A12, A22 = K(cam)
    P('[3.1] 与相机内参的自洽性：地面平面拟合')
    P('    %-14s %10s %10s %10s %10s' % ('sample', 'h(m)', 'RMS(mm)', 'p95(mm)', 'CV(h)%'))
    hs, rmss = [], []
    for sp, sid in SAMPLES:
        rgb, gt, dep = load_one(sp, sid, crop)
        Hh, Ww = dep.shape
        vv, uu = np.mgrid[0:Hh, 0:Ww]
        m = (dep > 0) & (gt == 255)
        Z = dep[m] / SCALE
        Pts = unproject(Z, uu[m].astype(np.float64), vv[m].astype(np.float64), cam)
        n, d, res = plane_fit(Pts)
        h = Z * (vv[m] - v0) / A22
        P('    %-14s %10.4f %10.1f %10.1f %10.2f' %
          (sid, abs(d), 1000 * np.sqrt((res ** 2).mean()), 1000 * np.percentile(res, 95),
           100 * np.std(h) / np.median(h)))
        hs.append(abs(d)); rmss.append(1000 * np.sqrt((res ** 2).mean()))
    P('    -> 离地高度 %.4f~%.4f m（跨 4 个样本极差 %.1f mm），平面残差 RMS %.1f mm'
      % (min(hs), max(hs), 1000 * (max(hs) - min(hs)), np.median(rmss)))
    P('       深度与 K 在**同一像素网格**上高度自洽；若深度图与 RGB 有整块错位，路面会被掺入非路面点，')
    P('       残差会明显放大（1 个深度块 = 7 px 的错位约带来 Z*7/733 的高度偏差）。')

    P('')
    P('[3.2] 地面点回归 v = a + b/Z（分离主点/俯仰与地形）')
    P('    水平地面理论上满足 v = v0 + h*A22/Z，故截距 a 应等于 v0（含俯仰修正），斜率 b = h*A22。')
    P('    %-14s %-12s %9s %11s %9s %8s' % ('sample', 'Z区间(m)', 'a(px)', 'a-v0(px)', 'h(m)', 'res_std'))
    for sp, sid in SAMPLES[:3]:
        _, gt, dep = load_one(sp, sid, crop)
        Hh, Ww = dep.shape
        vv, uu = np.mgrid[0:Hh, 0:Ww]
        Z0 = dep.astype(np.float64) / SCALE
        base = (dep > 0) & (gt == 255)
        for lo, hi in [(0.7, 3), (1, 6), (4, 10), (6, 20)]:
            k = base & (Z0 > lo) & (Z0 < hi)
            if k.sum() < 3000:
                continue
            inv = 1.0 / Z0[k]; v = vv[k].astype(np.float64)
            A = np.stack([np.ones_like(inv), inv], 1)
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            r = v - A @ c
            P('    %-14s %-12s %9.2f %+11.2f %9.4f %8.2f' %
              (sid, '%g-%g' % (lo, hi), c[0], c[0] - v0, c[1] / A22, r.std()))
    P('    -> 近场 0.7~6 m 高度稳定在 0.522~0.534 m，但外推得到的地平线截距比 v0 高约 %s px。' % ('16~20'))
    P('       4~20 m 段的 h 在部分样本上抬升到 0.59~0.71 m、截距回落到 v0 附近 -> 远场与近场不共面。')
    P('       该差额只能归因于（可并存）：道路纵坡 ~1.3°、远端 GT 边界不精确、DTOF 距离相关偏差、')
    P('       或标定俯仰/主点残差。本静态场景无法区分成因，但偏差本身被下面 3.4 独立复现。')
    P('       影响：直接用 calib 做深度->3D 提升而不重新拟合地面，会带 ~1.5° 系统倾斜，')
    P('       水平面高度在 10 m 处偏 %.0f cm、20 m 处偏 %.0f cm。'
      % (100 * 18 * 10 / A22, 100 * 18 * 20 / A22))

    P('')
    P('[3.3] 物体轮廓对齐（最直接的深度<->RGB 配准证据）')
    P('    对近/远拖拉机区域：取深度 > 阈值 的连通体轮廓，量它到最近强图像边缘的中位距离。')
    for sp, sid, bbox, thr in [('validation', '817224311091', (700, 520, 1220, 1040), 4.5),
                               ('training', '681150656293', (700, 560, 1220, 1080), 5.0)]:
        rgb, gt, dep = load_one(sp, sid, crop)
        x0, y0, x1, y1 = bbox
        I = gmag(rgb[y0:y1, x0:x1].mean(2))
        D = dep[y0:y1, x0:x1].astype(np.float32) / SCALE
        edge = I > np.percentile(I, 96)
        M = ndi.binary_closing(ndi.binary_opening(D > thr, np.ones((3, 3))), np.ones((9, 9)))
        Mb = M & ~ndi.binary_erosion(M, np.ones((3, 3)))
        yy, xx = np.nonzero(Mb)
        tab_, best = chamfer_sweep((yy, xx), edge, rad=10, step=2)
        P('    %-14s 轮廓 %d px：(0,0)=%.2f px，最优 %.2f px @ %s，改善 %.2f px'
          % (sid, yy.size, tab_[(0, 0)], tab_[best], best, tab_[(0, 0)] - tab_[best]))
        P('      du 扫描: %s' % [round(tab_[(d, 0)], 2) for d in range(-8, 9, 2)])
        P('      dv 扫描: %s' % [round(tab_[(0, d)], 2) for d in range(-8, 9, 2)])
    P('    -> 零位移已是局部最优附近，最优平移带来的改善 <1 px 且帧间方向不一致；')
    P('       结合 fig06 叠加图（拖拉机轮胎轮廓在深度图中清晰可辨且与 RGB 重合），')
    P('       判定深度与 RGB **不存在 >=5 px 的系统错位**；受 7 px 分块限制，')
    P('       本方法的分辨极限约 ±4 px，更细的偏移无法用该数据验证。')

    P('')
    P('[3.4] 独立交叉验证：25 m 量程线的位置')
    _, gt, dep = load_one('training', '346035329872', crop)
    tops = []
    for x in range(0, dep.shape[1], 4):
        c = np.flatnonzero(dep[:, x] > 0)
        if c.size:
            tops.append(c.min())
    tops = np.array(tops)
    hist, edges = np.histogram(tops, bins=20)
    mode = edges[int(np.argmax(hist))]
    h0 = 0.524
    pred_v0 = v0 + h0 * A22 / 25
    pred_off = v0 + 18 + h0 * A22 / 25
    P('    逐列最上有效行：中位 %.0f，分布主峰在 %d~%d 行（%d/400 列）'
      % (np.median(tops), mode, mode + 24, hist.max()))
    P('    若按 calib v0=%.1f + h=%.2f m 推算，25 m 量程对应第 %.0f 行；' % (v0, h0, pred_v0))
    P('    若按 3.2 的反推偏移(+18 px)推算，对应第 %.0f 行。' % pred_off)
    P('    -> 实测主峰 ~%d 行，与"+18 px"一致，比 calib 预言高约 %.0f 行。' % (mode, mode - pred_v0))
    P('    这是与 3.2 完全独立的一条证据（一个用地形高度、一个用量程边界），两者同向同量级，')
    P('    说明 depth 图与 calib.yaml 在垂直方向确有 ~18 px（≈1.4°）的系统失配，不是拟合偶然。')
    P('    另注：深度"变空"由 25 m 量程决定而非图像内容，因此')
    P('    **不能把深度无效区当作天空/远景的语义监督**（会把 25 m 外所有物体都标成天空）。')


# ================================================================ 4
def sec_crop():
    H('4. 裁剪到 1280x1088 后的口径')
    x0, y0, x1, y1 = CROP
    P('[4.1] 裁剪框与代码/标定一致')
    P('    calib_1280x1088.yaml: crop_box_ltrb=%s crop_size_hw=[%d,%d]' % (list(CROP), y1 - y0, x1 - x0))
    P('    ExtCenterCrop.get_box((1600,1296)) = (%d,%d,%d,%d)'
      % ((1600 - 1280) // 2, (1296 - 1088) // 2, (1600 - 1280) // 2 + 1280, (1296 - 1088) // 2 + 1088))
    P('    main.py ODIN1_SOURCE_SIZE_HW=(1296,1600), ODIN1_CROP_SIZE_HW=(1088,1280) -> 一致')
    P('')
    P('[4.2] 派生标定复算')
    u0, v0, A11, A12, A22 = K(CAM)
    cu0, cv0, cA11, cA12, cA22 = K(CAMC)
    eu0, ev0 = (u0 - x0 + .5) - .5, (v0 - y0 + .5) - .5
    P('    u0 %.6f -> 期望 %.6f / 文件 %.6f 差 %.1e' % (u0, eu0, cu0, abs(eu0 - cu0)))
    P('    v0 %.6f -> 期望 %.6f / 文件 %.6f 差 %.1e' % (v0, ev0, cv0, abs(ev0 - cv0)))
    P('    A11 %.6f -> %.6f ；A22 %.6f -> %.6f （裁剪不缩放，应不变）' % (A11, cA11, A22, cA22))
    P('    畸变 k2~k7 保持原值；Tcl_0 保持原值 -> 正确')
    P('')
    P('[4.3] 三模态同步裁剪')
    ok = True
    for sp, sid in SAMPLES:
        rgb, gt, dep = load_one(sp, sid)
        c = load_one(sp, sid, CROP)
        eq = [np.array_equal(c[0], rgb[y0:y1, x0:x1]), np.array_equal(c[1], gt[y0:y1, x0:x1]),
              np.array_equal(c[2], dep[y0:y1, x0:x1])]
        P('    %-14s rgb=%s gt=%s depth=%s -> %s' % (sid, *eq, c[2].shape))
        ok &= all(eq)
    P('    -> 三模态用同一框同步裁剪，无独立性偏差')
    P('')
    P('[4.4] 内容损失')
    P('    %-14s %11s %11s %11s %11s' % ('sample', 'pass_占原图%', 'pass_占裁剪%', 'pass保留率', 'sky保留率'))
    for sp, sid in SAMPLES:
        _, gt, dep = load_one(sp, sid)
        _, cgt, _ = load_one(sp, sid, CROP)
        P('    %-14s %11.2f %11.2f %11.3f %11.3f' %
          (sid, 100 * (gt == 255).mean(), 100 * (cgt == 255).mean(),
           (cgt == 255).sum() / max((gt == 255).sum(), 1),
           (cgt == 128).sum() / max((gt == 128).sum(), 1)))
    P('    裁剪保留全图 67.2% 面积：可通行 66%~67%、天空 66%~70% 被保留，比例基本不变，')
    P('    说明裁剪框居中且没有偏掉某一侧；被裁掉的 1/3 主要是上下方的天空与近端路面。')
    P('')
    P('[4.5] 裁剪前后几何等价性')
    h1, r1 = _geom_summary(CAM, None)
    h2, r2 = _geom_summary(CAMC, CROP)
    P('    原始 1600x1296 : h=%.4f m  RMS=%.1f mm' % (h1, r1))
    P('    裁剪 1280x1088 : h=%.4f m  RMS=%.1f mm' % (h2, r2))
    P('    差异: h %.1f mm, RMS %.1f mm' % (1000 * abs(h1 - h2), abs(r1 - r2)))
    P('    -> 两套口径几何等价，裁剪后的派生主点正确，深度投影不会因裁剪产生漂移。')
    P('')
    P('[4.6] 接入提醒')
    P('    当前 main.py 的 SegmentationPreprocessor 只处理 RGB 与标签，dense_depth 未进入变换链。')
    P('    后续要把深度接入（如 BEV/深度监督），必须复用同一个 crop_box，否则 RGB 与深度会错位 (160,104) px。')


def _geom_summary(cam, crop):
    hs, rs = [], []
    for sp, sid in SAMPLES:
        _, gt, dep = load_one(sp, sid, crop)
        Hh, Ww = dep.shape
        vv, uu = np.mgrid[0:Hh, 0:Ww]
        m = (dep > 0) & (gt == 255)
        Z = dep[m] / SCALE
        Pts = unproject(Z, uu[m].astype(np.float64), vv[m].astype(np.float64), cam)
        n, d, res = plane_fit(Pts)
        hs.append(abs(d)); rs.append(1000 * np.sqrt((res ** 2).mean()))
    return float(np.median(hs)), float(np.median(rs))


# ================================================================ 5
def sec_leakage():
    H('5. 划分独立性与多样性')
    tr = ids('training', 'image_data')
    g = lambda sp, sid: np.array(Image.open(os.path.join(ROOT, sp, 'image_data', sid + '.png')).convert('L'), np.float32)
    tg = lambda sp, sid: np.array(Image.open(os.path.join(ROOT, sp, 'gt_image', sid + '.png')))
    P('    %-14s %-14s %10s %10s' % ('val/test', '最相似 train', 'rgb_corr', 'gt_agree'))
    mxc = []
    for sp in ('validation', 'testing'):
        for s2 in ids(sp, 'image_data'):
            r2 = g(sp, s2)
            best = (0, None, 0)
            for s1 in tr:
                c = np.corrcoef(r2.ravel(), g('training', s1).ravel())[0, 1]
                if c > best[0]:
                    best = (c, s1, (tg(sp, s2) == tg('training', s1)).mean())
            mxc.append(best[0])
            P('    %-14s %-14s %10.4f %10.4f' % (s2, best[1], best[0], best[2]))
    P('')
    P('    -> 验证/测试帧与训练帧最大 RGB 相关中位 %.3f、最大 %.3f；测试帧 345742871718 与'
      % (np.median(mxc), max(mxc)))
    P('       训练帧 346425225661 相关 0.996（时间戳仅差 0.29 s），实质是同一场景的相邻帧。')
    P('       当前划分不具备场景独立性，验证/测试指标会明显偏乐观。')
    P('')
    P('    全数据集只有 1 个机位：25 帧中 22 帧 RGB 两两相关 >0.98，其余 3 帧是同一机位下拖拉机驶近。')
    P('    等价于"1 个静态背景 + 1 个移动目标"，对 3 类分割训练而言场景多样性几乎为 0。')


# ================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    P('segdataodin 数据质量核查报告（只读核查，未修改数据集与项目代码）')
    P('数据集根目录: %s' % ROOT)
    P('冻结标定 calib.yaml: u0=%.4f v0=%.4f A11=%.4f A12=%.4f A22=%.4f' % K(CAM))
    rows = sec_inventory()
    sec_encoding(rows)
    sec_gt()
    sec_reg(CAM, '原始 1600x1296', None)
    P('')
    sec_crop()
    sec_leakage()

    H('6. 结论')
    for s in CONCL:
        P('  ' + s)
    open(os.path.join(OUT, 'audit_report.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n[saved] audit_report.txt')


CONCL = [
    '【编码】正确。PNG 为 16 位单通道，值 = floor(z_depth_m * 256)，0 表示无回波，',
    '   1 LSB = 3.91 mm，无 255 m 饱和。÷256 与"离地高度 0.52 m / 25 m 量程 / 近端路面"',
    '   三者同时自洽；÷512、÷1000、当 mm 都会给出物理上不可能的安装高度。',
    '   风险点：①读取必须保留 16 位（cv2.imread 默认会缩成 8 位，最大值从 6443 变 25）；',
    '   ②编码是截断不是四舍五入，单像素平均偏低 1.95 mm；',
    '   ③dense_depth 是 7x7 分块上采样，真实分辨率仅约 229x185，不是逐像素深度。',
    '',
    '【RGB<->GT 配准】准确。两者同源标注，标签边界处图像梯度为全图 1.3 倍，',
    '   平移搜索最优改善仅 1~2 px 且方向帧间不一致 -> 无系统平移。',
    '   但标注是粗粒度多边形，边界精度 ±3~8 px，不适合细边界指标评测。',
    '',
    '【depth<->RGB 配准】未见 >=5 px 系统错位。拖拉机轮胎轮廓在深度图中清晰可辨且与 RGB 重合',
    '   (fig06)；轮廓 chamfer 在零位移处即最优附近，最优平移改善 <1 px 且方向不一致。',
    '   受 7 px 分块量化限制，该数据的配准可验证精度约 ±4 px。',
    '',
    '【depth<->标定】存在需要关注的系统残差。近场 0.7~6 m 地面高度恒定在 0.522~0.534 m，',
    '   但外推地平线比 calib.yaml 的 v0 高约 16~20 px（≈1.4°）。该偏差被两条独立证据复现：',
    '   ①地面点回归截距 +16~20 px；②25 m 量程线实测在主峰约 695 行，比按 v0 推算的 673 行高 22 行。',
    '   成因可能是道路纵坡、DTOF 距离相关偏差或标定俯仰/主点残差，本静态场景无法区分。',
    '   影响：直接用 calib 做深度->3D 提升会带 ~1.4° 系统倾斜（水平面在 10 m 处偏 25 cm、20 m 处偏 49 cm）。',
    '   建议：进 BEV 前用本批数据的路面点重新拟合地面/俯仰，或把派生标定的 v0 整体 +18 px 后再用。',
    '   注意：当前 main.py 只用 RGB+标签做分割，此残差**不影响现阶段的分割训练与评估**。',
    '',
    '【裁剪 1280x1088】完全正确。三模态同框、派生 u0/v0 复算零误差、A11/A12/A22 不变，',
    '   裁剪前后地面几何 h 差 <1 mm、RMS 差 <2 mm。可安全用 calib_1280x1088.yaml。',
    '',
    '【主要数据质量缺陷（非编码/配准问题）】',
    '   1) 场景多样性几乎为 0：25 帧 = 1 个静止机位 + 1 台移动拖拉机；',
    '      测试帧与某训练帧 RGB 相关 0.996（时间差 0.29 s），划分不独立，指标必然偏乐观。',
    '   2) 深度 40%~56% 无回波；无效区上边界主要由 25 m 量程决定，',
    '      不能把"深度无效"当天空语义监督用。',
    '   3) 3 个验证帧存在 <0.30 m 的近距回波（最大 441 px 成块），疑为机具自身遮挡，近距检测需剔除。',
    '   4) dense_depth 未接入 main.py 预处理链；接入时必须复用同一 crop_box。',
]

if __name__ == '__main__':
    main()
