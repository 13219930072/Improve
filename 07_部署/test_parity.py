# -*- coding: utf-8 -*-
"""
===========================================================================
test_parity.py — LAMF GPU 版 vs CPU 版的数值对齐验证
===========================================================================
在同一权重、同一输入下对比两条推理路径的输出：
  GPU 版: net.py + utils/lamf_mamba_layer.py (mamba_ssm CUDA 算子)
  CPU 版: deploy/lamf_cpu.py (纯 PyTorch)

验收标准:
  - 与官方 PyTorch 参考实现: bit-exact (selective scan 层面)
  - GPU(CUDA算子) vs CPU: 相对误差 < 1e-2
    （CUDA 融合算子本身有浮点差异，实测单模块 ~1.9e-2 相对；
      真正的正确性基准是 PyTorch 参考实现，见 --strict 模式）

用法:
  python deploy/test_parity.py                 # 端到端对比 (需 GPU)
  python deploy/test_parity.py --strict        # 与官方参考实现 bit-exact 校验
===========================================================================
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

CKPT = os.path.join(_PROJ_ROOT, 'experiments', 'LAMF_Full', 'checkpoints',
                    'LAMF_Full_Final_07-17-13-19.pth')
TEST_ROOT = '/home/hyhz/HT/test_img'


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in sd.items()}


def load_gpu_model(device='cuda'):
    """GPU 版：原版网络（CUDA Mamba）"""
    from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
    from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
    ckpt = torch.load(CKPT, map_location=device)
    E = Restormer_Encoder().to(device); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = Restormer_Decoder().to(device); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = LocalEnhancedVSSBlock(dim=64).to(device)
    BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = DetailFeatureExtraction(num_layers=1).to(device)
    DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def read_pair(ds, img_id):
    from utils.img_read_save import image_read_cv2
    d_ir = os.path.join(TEST_ROOT, ds, 'ir')
    d_vi = os.path.join(TEST_ROOT, ds, 'vi')
    f_ir = next(os.path.join(d_ir, f) for f in os.listdir(d_ir)
                if os.path.splitext(f)[0] == img_id)
    f_vi = next(os.path.join(d_vi, f) for f in os.listdir(d_vi)
                if os.path.splitext(f)[0] == img_id)
    ir = image_read_cv2(f_ir, 'GRAY')
    vi = image_read_cv2(f_vi, 'GRAY')
    return ir, vi


def run_gpu(model_parts, ir_np, vis_np, device='cuda'):
    E, D, BF, DF = model_parts
    ir = torch.FloatTensor(ir_np[np.newaxis, np.newaxis, ...] / 255.0).to(device)
    vis = torch.FloatTensor(vis_np[np.newaxis, np.newaxis, ...] / 255.0).to(device)
    with torch.no_grad():
        fV_B, fV_D, _ = E(vis)
        fI_B, fI_D, _ = E(ir)
        fF_B = BF(fI_B + fV_B)
        fF_D = DF(fI_D + fV_D)
        fused, _ = D(vis, fF_B, fF_D)
    return fused


def run_cpu(model, ir_np, vis_np):
    ir = torch.FloatTensor(ir_np[np.newaxis, np.newaxis, ...] / 255.0)
    vis = torch.FloatTensor(vis_np[np.newaxis, np.newaxis, ...] / 255.0)
    with torch.no_grad():
        return model(ir, vis)


def strict_scan_check():
    """与官方 selective_scan_ref 的 bit-exact 校验"""
    from mamba_cpu import selective_scan_cpu
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    torch.manual_seed(0)
    B, D, L, N = 1, 128, 512, 16
    u = torch.randn(B, D, L)
    dt = torch.rand(B, D, L) * 0.1
    A = -torch.rand(D, N)
    Bm = torch.randn(B, N, L)
    Cm = torch.randn(B, N, L)
    Dp = torch.randn(D)
    z = torch.randn(B, D, L)
    ref = selective_scan_ref(u, dt, A, Bm, Cm, D=Dp, z=z, delta_softplus=True)
    mine = selective_scan_cpu(u, dt, A, Bm, Cm, D=Dp, z=z, delta_softplus=True)
    d = (ref - mine).abs().max().item()
    print(f"selective_scan 纯 PyTorch 实现 vs 官方参考: max diff = {d:.3e}")
    print("✅ bit-exact" if d == 0 else "⚠️ 存在差异")
    return d == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strict', action='store_true', help='仅做 selective scan bit-exact 校验')
    ap.add_argument('--pairs', nargs='+', default=['TNO:01', 'RoadScene:FLIR_06392', 'M3FD:00000'])
    args = ap.parse_args()

    if args.strict:
        ok = strict_scan_check()
        sys.exit(0 if ok else 1)

    print("=" * 70)
    print("LAMF CPU/GPU 数值对齐测试")
    print("=" * 70)

    # 先做严格的 scan 层校验
    strict_scan_check()
    print()

    from lamf_cpu import LAMF_CPU
    print("构建 CPU 版模型...")
    cpu_model = LAMF_CPU()
    cpu_model.eval()
    cpu_model.load_lamf_weights(CKPT, verbose=False)
    print("✅ CPU 版就绪")

    print("\n构建 GPU 版模型...")
    gpu_parts = load_gpu_model()
    print("✅ GPU 版就绪\n")

    print(f"{'图像':<24} {'最大绝对差':>12} {'相对误差':>12} {'GPU耗时':>10} {'CPU耗时':>10}")
    print("-" * 72)
    all_rel = []
    for pair in args.pairs:
        ds, img_id = pair.split(':')
        ir_np, vi_np = read_pair(ds, img_id)

        t0 = time.time()
        y_gpu = run_gpu(gpu_parts, ir_np, vi_np).cpu()
        t_gpu = time.time() - t0

        t0 = time.time()
        y_cpu = run_cpu(cpu_model, ir_np, vi_np)
        t_cpu = time.time() - t0

        d = (y_gpu - y_cpu).abs()
        max_abs = d.max().item()
        rel = max_abs / max(y_gpu.abs().max().item(), 1e-8)
        all_rel.append(rel)
        print(f"{pair:<24} {max_abs:>12.3e} {rel:>12.3e} {t_gpu*1000:>9.0f}ms {t_cpu*1000:>9.0f}ms")

    print("-" * 72)
    worst = max(all_rel)
    print(f"\n最差相对误差: {worst:.3e}")
    if worst < 1e-2:
        print("✅ 通过（差异在 CUDA 融合算子的浮点误差范围内）")
    else:
        print("❌ 未通过，需排查（检查 CPU 实现是否与参考实现一致）")
        sys.exit(1)


if __name__ == '__main__':
    main()
