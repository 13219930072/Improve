# -*- coding: utf-8 -*-
"""
===========================================================================
infer.py — LAMF 融合推理（CPU / ARM 设备）
===========================================================================
用法:
  # 单张
  python infer.py --ir path/to/ir.png --vis path/to/vis.png --out fused.png

  # 批量（目录，文件名按去扩展名匹配）
  python infer.py --ir-dir data/ir --vis-dir data/vi --out-dir results

  # 指定分辨率缩放（香橙派上推荐 128 或 256 以控制耗时）
  python infer.py --ir a.png --vis b.png --out c.png --size 128

输出: 融合图像 PNG（与训练/评估管线口径一致：min-max → ×255 → uint8）
===========================================================================
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from lamf_cpu import LAMF_CPU  # noqa: E402

DEFAULT_CKPT = os.path.join(_HERE, 'weights', 'lamf_final.pth')


def read_gray(path):
    """读取灰度图，返回 float32 [0,255] 的 2D 数组"""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    return np.round(cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2GRAY))


def resize_if_needed(arr, size):
    """按长边缩放到 size（保持宽高比，且为偶数），size=None 则不缩放"""
    if size is None:
        return arr
    import cv2
    h, w = arr.shape[:2]
    if max(h, w) <= size:
        return arr
    scale = size / max(h, w)
    nh, nw = int(round(h * scale / 2) * 2), int(round(w * scale / 2) * 2)
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)


def fuse(model, ir_np, vi_np):
    """ir/vis: float32 [0,255] 2D 数组 → 融合图 uint8 [0,255]"""
    ir = torch.FloatTensor(ir_np[np.newaxis, np.newaxis, ...] / 255.0)
    vis = torch.FloatTensor(vi_np[np.newaxis, np.newaxis, ...] / 255.0)
    t0 = time.time()
    with torch.no_grad():
        out = model(ir, vis)
    elapsed = time.time() - t0
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    fused = np.squeeze((out * 255.0).numpy()).astype(np.uint8)
    return fused, elapsed


def save_gray(arr, path):
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, arr)
    return path


def main():
    ap = argparse.ArgumentParser(description='LAMF 融合推理 (CPU)')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help='权重路径')
    ap.add_argument('--ir', help='单张红外图像路径')
    ap.add_argument('--vis', help='单张可见光图像路径')
    ap.add_argument('--out', help='单张输出路径')
    ap.add_argument('--ir-dir', help='批量：红外目录')
    ap.add_argument('--vis-dir', help='批量：可见光目录')
    ap.add_argument('--out-dir', help='批量：输出目录')
    ap.add_argument('--size', type=int, default=None,
                    help='缩放长边到该尺寸（推荐 128/256，None=原生分辨率）')
    ap.add_argument('--threads', type=int, default=0, help='CPU 线程数（0=自动）')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f'加载模型: {args.ckpt}')
    model = LAMF_CPU()
    model.eval()
    model.load_lamf_weights(args.ckpt, verbose=False)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'✅ LAMF 就绪 ({n_param/1e6:.4f}M 参数, CPU 推理)')

    # ── 单张模式 ──
    if args.ir and args.vis:
        if not args.out:
            args.out = 'fused.png'
        ir_np = resize_if_needed(read_gray(args.ir), args.size)
        vi_np = resize_if_needed(read_gray(args.vis), args.size)
        fused, t = fuse(model, ir_np, vi_np)
        save_gray(fused, args.out)
        print(f'✅ {args.out}  ({ir_np.shape[1]}×{ir_np.shape[0]}, {t:.2f}s)')
        return

    # ── 批量模式 ──
    if not (args.ir_dir and args.vis_dir and args.out_dir):
        ap.error('请指定 --ir/--vis/--out（单张）或 --ir-dir/--vis-dir/--out-dir（批量）')

    os.makedirs(args.out_dir, exist_ok=True)
    ir_files = sorted(os.listdir(args.ir_dir))
    total_t, n = 0.0, 0
    print(f'批量处理 {len(ir_files)} 对图像 → {args.out_dir}')
    for fn in ir_files:
        stem = os.path.splitext(fn)[0]
        # 按去扩展名匹配可见光图像
        vis_fn = next((f for f in os.listdir(args.vis_dir)
                       if os.path.splitext(f)[0] == stem), None)
        if vis_fn is None:
            print(f'  ⚠ 跳过 {fn}（可见光目录中无匹配）')
            continue
        ir_np = resize_if_needed(read_gray(os.path.join(args.ir_dir, fn)), args.size)
        vi_np = resize_if_needed(read_gray(os.path.join(args.vis_dir, vis_fn)), args.size)
        fused, t = fuse(model, ir_np, vi_np)
        save_gray(fused, os.path.join(args.out_dir, stem + '.png'))
        total_t += t
        n += 1
        if n % 10 == 0 or n == len(ir_files):
            print(f'  {n}/{len(ir_files)}  平均 {total_t/n:.2f}s/张')
    print(f'✅ 完成 {n} 张，总耗时 {total_t:.1f}s，平均 {total_t/max(n,1):.2f}s/张')


if __name__ == '__main__':
    main()
