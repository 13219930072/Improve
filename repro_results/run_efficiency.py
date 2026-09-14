# -*- coding: utf-8 -*-
"""
===========================================================================
run_efficiency.py — 统一效率评估（按交接文档 §9-§14 规范）
===========================================================================
输出文件（每方法独立目录）:
  <Method>/outputs/runtime/params.csv       (组件级 Params: total/trainable, Encoder/Fusion/Decoder)
  <Method>/outputs/runtime/flops.csv        (128/256/512: FLOPs, MACs, Params)
  <Method>/outputs/runtime/latency.csv      (128/256/512: mean/std latency_ms, peak_memory_MB, FPS)
  <Method>/outputs/runtime/throughput.csv   (同信息, FPS 视角)

用法:
  python repro_results/run_efficiency.py --method lamf
  python repro_results/run_efficiency.py --method cddfuse

测量协议:
  batch=1, single GPU, CUDA
  warmup=30, repeats=100, torch.cuda.synchronize()
  FLOPs: thop 0.1.1, 输入 [1,1,H,W]
===========================================================================
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT_BASE = os.path.dirname(os.path.abspath(__file__))
WARMUP = 30
REPEATS = 100
RESOLUTIONS = [128, 256, 512]


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def build_wrapper(E, D, BF, DF):
    """组合成单一 forward: (ir, vis) → fused (与评估路径一致)"""
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.E, self.D = E, D
            self.BF, self.DF = BF, DF

        def forward(self, ir, vis):
            fV_B, fV_D, _ = self.E(vis)
            fI_B, fI_D, _ = self.E(ir)
            fF_B = self.BF(fI_B + fV_B)
            fF_D = self.DF(fI_D + fV_D)
            fused, _ = self.D(vis, fF_B, fF_D)
            return fused

    return FullModel().to(DEVICE)


def count_params(modules, names):
    rows = []
    for name, m in zip(names, modules):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rows.append({"Component": name, "Total_Params": total, "Trainable_Params": trainable})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['lamf', 'cddfuse'], required=True)
    args = ap.parse_args()
    mkey = args.method

    if mkey == "lamf":
        from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
        from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
        ckpt = torch.load("/home/hyhz/HT/code/MY-CDDFuse/experiments/LAMF_Full/checkpoints/LAMF_Full_Final_07-17-13-19.pth", map_location=DEVICE)
        E = Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
        D = Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
        BF = LocalEnhancedVSSBlock(dim=64).to(DEVICE); BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
        DF = DetailFeatureExtraction(num_layers=1).to(DEVICE); DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
        comp_names = ["Encoder", "BaseFuse(LESS-Mamba)", "DetailFuse(NAFDetail)", "Decoder"]
    else:
        cdd_dir = "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main"
        sys.path.insert(0, cdd_dir)
        import net as cdd_net
        ckpt = torch.load("/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main/models/CDDFuse_IVF.pth", map_location=DEVICE)
        E = cdd_net.Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
        D = cdd_net.Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
        BF = cdd_net.BaseFeatureExtraction(dim=64, num_heads=8).to(DEVICE)
        BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
        DF = cdd_net.DetailFeatureExtraction(num_layers=1).to(DEVICE)
        DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
        sys.path.remove(cdd_dir)
        comp_names = ["Encoder", "BaseFuse(Transformer)", "DetailFuse(INN)", "Decoder"]

    print(f"✅ {mkey} 加载完成")

    out_dir = os.path.join(OUT_BASE, mkey, "outputs", "runtime")
    os.makedirs(out_dir, exist_ok=True)

    model = build_wrapper(E, D, BF, DF)
    model.eval()

    # ── 1. params.csv ──
    param_rows = count_params([E, BF, DF, D], comp_names)
    total = sum(r["Total_Params"] for r in param_rows)
    trainable = sum(r["Trainable_Params"] for r in param_rows)
    param_rows.append({"Component": "TOTAL", "Total_Params": total, "Trainable_Params": trainable})
    with open(os.path.join(out_dir, "params.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Component", "Total_Params", "Trainable_Params"])
        w.writeheader()
        w.writerows(param_rows)
    print(f"✅ params.csv (TOTAL={total/1000:.1f}K)")

    # ── 2-4. FLOPs / Latency / Memory / FPS per resolution ──
    from thop import profile
    from thop import clever_format

    flops_rows, lat_rows, fps_rows = [], [], []
    for H in RESOLUTIONS:
        ir = torch.randn(1, 1, H, H).to(DEVICE)
        vis = torch.randn(1, 1, H, H).to(DEVICE)

        # 峰值显存（前向一帧, batch=1）
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model(ir, vis)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

        # FLOPs 测量（使用次元 profile 含前向? 用 profile(verbose=False)）
        with torch.no_grad():
            flops, params = profile(model, inputs=(ir, vis), verbose=False)

        # Latency: warmup + repeats
        with torch.no_grad():
            for _ in range(WARMUP):
                model(ir, vis)
            torch.cuda.synchronize()
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                model(ir, vis)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        lat_mean = float(np.mean(times))
        lat_std = float(np.std(times))
        fps = 1000.0 / lat_mean

        flops_rows.append({"Input": f"{H}x{H}", "N": H * H, "FLOPs": flops, "MACs": flops / 2,
                          "Params": params, "Tool": "thop 0.1.1", "Batch": 1, "Status": "MEASURED"})
        lat_rows.append({"Input": f"{H}x{H}", "mean_latency_ms": lat_mean, "std_latency_ms": lat_std,
                         "peak_memory_MB": peak_mb, "FPS": fps,
                         "warmup": WARMUP, "repeats": REPEATS, "batch": 1})
        fps_rows.append({"Input": f"{H}x{H}", "FPS": fps, "latency_ms": lat_mean})
        print(f"  {H}x{H}: FLOPs={flops/1e9:.2f}G, Latency={lat_mean:.1f}ms±{lat_std:.2f}, Mem={peak_mb:.1f}MB, FPS={fps:.1f}")

    for name, rows, cols in [
        ("flops.csv", flops_rows, ["Input", "N", "FLOPs", "MACs", "Params", "Tool", "Batch", "Status"]),
        ("latency.csv", lat_rows, ["Input", "mean_latency_ms", "std_latency_ms", "peak_memory_MB", "FPS", "warmup", "repeats", "batch"]),
        ("throughput.csv", fps_rows, ["Input", "FPS", "latency_ms"]),
    ]:
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print(f"✅ flops.csv / latency.csv / throughput.csv 已生成 → {out_dir}")

    # ── 专项: Base 融合层对比（交接文档 §19.1）—— 参数 + FLOPs（128×128）──
    # Base 融合层输入: 融合前的 Base 特征. LAMF: LocalEnhancedVSSBlock 吃 [B,H,W,64](NHWC);
    # CDDFuse: BaseFeatureExtraction 吃 [B,64,H,W](NCHW). 统一包装为 128×128.
    with torch.no_grad():
        # 两个 Base 融合层均接受 NCHW [B,C,H,W] （LAMF 块内部自行 permute 到 NHWC）
        x = torch.randn(1, 64, 128, 128).to(DEVICE)
        flops_b, _ = profile(BF, inputs=(x,), verbose=False)
    base_rows = [{"Component": "BaseFuse", "Params": sum(p.numel() for p in BF.parameters()),
                  "FLOPs_128x128": flops_b, "Tool": "thop 0.1.1"}]
    with open(os.path.join(out_dir, "base_fuse_params.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Component", "Params", "FLOPs_128x128", "Tool"])
        w.writeheader()
        w.writerows(base_rows)
    print("✅ base_fuse_params.csv（专项: Base 融合层参数 + FLOPs）")


if __name__ == '__main__':
    main()
