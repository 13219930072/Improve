# -*- coding: utf-8 -*-
"""
===========================================================================
run_full_eval.py — LAMF / CDDFuse 统一全量评估（按交接文档 §6-§8 规范）
===========================================================================
对 4 数据集 655 对图像: 融合 → 保存 PNG → float32 读回 → 8 指标 → CSV
输出（每方法独立目录）:
  <Method>/outputs/fused_images/<Dataset>/<Image_ID>.png
  <Method>/outputs/metrics/per_image.csv        (Image_ID, Dataset, 8指标, inference_time_ms, peak_memory_MB)
  <Method>/outputs/metrics/average.csv          (Dataset, N, mean/std/median/min/max 8指标 + ALL 行)
  <Method>/outputs/runtime/dataset_runtime.csv  (Method, Dataset, N, Total_Inference_Time, Average_Inference_Time)
  <Method>/experiment_meta.json

用法:
  python repro_results/run_full_eval.py --method lamf
  python repro_results/run_full_eval.py --method cddfuse
===========================================================================
"""
import os
import sys
import csv
import json
import time
import argparse
import hashlib
from collections import defaultdict

import numpy as np
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_ROOT = "/home/hyhz/HT/test_img"
OUT_BASE = os.path.dirname(os.path.abspath(__file__))

# 权重（正确版，与论文数据一致）
CKPTS = {
    "lamf": {
        "path": "/home/hyhz/HT/code/MY-CDDFuse/experiments/LAMF_Full/checkpoints/LAMF_Full_Final_07-17-13-19.pth",
        "source": "experiments/LAMF_Full/checkpoints/LAMF_Full_Final_07-17-13-19.pth (自训练, 与本服务器2026-07-17实验E一致)",
        "network": "LAMF (E): LESS-Mamba + NAFDetailBlock",
    },
    "cddfuse": {
        "path": "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main/models/CDDFuse_IVF.pth",
        "source": "CDDFuse 官方仓库 MMIF-CDDFuse-main/models/CDDFuse_IVF.pth (官方复现库内置权重)",
        "network": "CDDFuse (CVPR 2023): Restormer + BaseFeatureExtraction + INN Detail",
    },
}

DATASETS = ["TNO", "RoadScene", "M3FD", "FMB"]
METRICS = ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def load_lamf():
    from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
    from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
    ckpt = torch.load(CKPTS["lamf"]["path"], map_location=DEVICE)
    E = Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = LocalEnhancedVSSBlock(dim=64).to(DEVICE); BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = DetailFeatureExtraction(num_layers=1).to(DEVICE); DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def load_cddfuse():
    cdd_dir = "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main"
    sys.path.insert(0, cdd_dir)  # 使 import net 命中 MMIF 的 net.py
    import net as cdd_net
    ckpt = torch.load(CKPTS["cddfuse"]["path"], map_location=DEVICE)
    E = cdd_net.Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = cdd_net.Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = cdd_net.BaseFeatureExtraction(dim=64, num_heads=8).to(DEVICE)
    BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = cdd_net.DetailFeatureExtraction(num_layers=1).to(DEVICE)
    DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    # 恢复 path 顺序，避免后续脚本 import 污染
    sys.path.remove(cdd_dir)
    return E, D, BF, DF


def find_img(ds, sub, img_id):
    d = os.path.join(TEST_ROOT, ds, sub)
    for fn in os.listdir(d):
        if os.path.splitext(fn)[0] == img_id:
            return os.path.join(d, fn)
    raise FileNotFoundError(f"{d}/{img_id}.* 不存在")


def fuse_one(E, D, BF, DF, ir, vis):
    with torch.no_grad():
        fV_B, fV_D, _ = E(vis)
        fI_B, fI_D, _ = E(ir)
        fF_B = BF(fI_B + fV_B)
        fF_D = DF(fI_D + fV_D)
        fused, _ = D(vis, fF_B, fF_D)
    return fused


def compute_metrics(fi_f, ir_f, vi_f):
    from utils.Evaluator import Evaluator
    return {
        "EN": Evaluator.EN(fi_f),
        "SD": Evaluator.SD(fi_f),
        "SF": Evaluator.SF(fi_f),
        "MI": Evaluator.MI(fi_f, ir_f, vi_f),
        "SCD": Evaluator.SCD(fi_f, ir_f, vi_f),
        "VIF": Evaluator.VIFF(fi_f, ir_f, vi_f),
        "Qabf": Evaluator.Qabf(fi_f, ir_f, vi_f),
        "SSIM": Evaluator.SSIM(fi_f, ir_f, vi_f),
    }


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['lamf', 'cddfuse'], required=True)
    args = ap.parse_args()

    mkey = args.method
    out_dir = os.path.join(OUT_BASE, mkey)
    fused_dir = os.path.join(out_dir, "outputs", "fused_images")
    metric_dir = os.path.join(out_dir, "outputs", "metrics")
    runtime_dir = os.path.join(out_dir, "outputs", "runtime")
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(runtime_dir, exist_ok=True)

    from utils.img_read_save import image_read_cv2, img_save

    if mkey == "lamf":
        E, D, BF, DF = load_lamf()
    else:
        E, D, BF, DF = load_cddfuse()
    print(f"✅ {mkey} 加载完成")

    rows = []
    runtime_rows = []
    t_start = time.time()

    for ds in DATASETS:
        ir_dir = os.path.join(TEST_ROOT, ds, "ir")
        ids = sorted(os.path.splitext(fn)[0] for fn in os.listdir(ir_dir))
        out_ds = os.path.join(fused_dir, ds)
        os.makedirs(out_ds, exist_ok=True)
        print(f"▶ {ds}: {len(ids)} 张")
        ds_time = 0.0
        for i, img_id in enumerate(ids):
            ir_f = image_read_cv2(find_img(ds, "ir", img_id), 'GRAY')
            vi_f = image_read_cv2(find_img(ds, "vi", img_id), 'GRAY')
            ir = torch.FloatTensor(ir_f[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
            vis = torch.FloatTensor(vi_f[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                fV_B, fV_D, _ = E(vis)
                fI_B, fI_D, _ = E(ir)
                fF_B = BF(fI_B + fV_B)
                fF_D = DF(fI_D + fV_D)
                fused, _ = D(vis, fF_B, fF_D)
            torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

            # 后处理: min-max → ×255 → uint8 → 保存 PNG → float32 读回（硬性要求）
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
            fi_np = np.squeeze((fused * 255.0).cpu().numpy()).astype(np.uint8)
            img_save(fi_np, img_id, out_ds)
            fi_f = image_read_cv2(os.path.join(out_ds, img_id + ".png"), 'GRAY')

            m = compute_metrics(fi_f, ir_f, vi_f)
            rows.append({"Image_ID": img_id, "Dataset": ds, **m,
                         "inference_time_ms": infer_ms, "peak_memory_MB": peak_mb})
            ds_time += infer_ms
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(ids)} ({infer_ms:.1f}ms)")
        runtime_rows.append({"Method": mkey, "Dataset": ds, "N": len(ids),
                             "Total_Inference_Time": round(ds_time / 1000.0, 3),
                             "Average_Inference_Time": round(ds_time / 1000.0 / len(ids), 3)})
        print(f"  ✅ {ds} 完成, 平均 {ds_time/1000.0/len(ids):.3f}s/张")

    # ── per_image.csv ──
    per_img_path = os.path.join(metric_dir, "per_image.csv")
    with open(per_img_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Image_ID", "Dataset"] + METRICS + ["inference_time_ms", "peak_memory_MB"])
        w.writeheader()
        w.writerows(rows)
    print(f"✅ per_image.csv: {len(rows)} 行")

    # ── average.csv（每数据集 + ALL，mean/std/median/min/max）──
    def write_average(records, ds_label, n, path):
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["Dataset", "Stat", "N"] + METRICS)
            for stat in ["mean", "std", "median", "min", "max"]:
                line = [ds_label, stat, n]
                for mk in METRICS:
                    vals = [r[mk] for r in records]
                    line.append(round(stats(vals)[stat], 8))
                w.writerow(line)

    avg_path = os.path.join(metric_dir, "average.csv")
    if os.path.exists(avg_path):
        os.remove(avg_path)
    all_records = []
    for ds in DATASETS:
        recs = [r for r in rows if r["Dataset"] == ds]
        write_average(recs, ds, len(recs), avg_path)
        all_records.extend(recs)
    write_average(all_records, "ALL", len(all_records), avg_path)
    print(f"✅ average.csv（含 ALL 655 行统计）")

    # ── dataset_runtime.csv ──
    rt_path = os.path.join(runtime_dir, "dataset_runtime.csv")
    with open(rt_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Dataset", "N", "Total_Inference_Time", "Average_Inference_Time"])
        w.writeheader()
        w.writerows(runtime_rows)
    print(f"✅ dataset_runtime.csv")

    # ── experiment_meta.json ──
    meta = {
        "method": mkey,
        "network": CKPTS[mkey]["network"],
        "year": "2023" if mkey == "cddfuse" else "2026",
        "repository": "https://github.com/Zhaozixiang1228/MMIF-CDDFuse" if mkey == "cddfuse" else "local (MY-CDDFuse, 本仓库)",
        "checkpoint": CKPTS[mkey]["path"],
        "checkpoint_sha256": sha256(CKPTS[mkey]["path"]),
        "checkpoint_source": CKPTS[mkey]["source"],
        "python": "3.8",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": "NVIDIA RTX 3090 24GB",
        "dataset_root": TEST_ROOT,
        "num_images": len(rows),
        "input_order": "(ir, vis) 各 [B,1,H,W], 共享Encoder",
        "input_range": "[0,1]",
        "padding": "none (原生分辨率直接前向, fully-convolutional)",
        "output_postprocess": "min-max → ×255 → uint8 → save PNG → float32 readback",
        "evaluator": "/home/hyhz/HT/code/MY-CDDFuse/utils/Evaluator.py (EN/SD/SF/MI/SCD/VIF/Qabf/SSIM)",
        "command": f"python repro_results/run_full_eval.py --method {mkey}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "experiment_meta.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ experiment_meta.json")

    # ── sanity check ──
    anomalies = []
    if len(rows) != 655:
        anomalies.append(f"N={len(rows)} != 655")
    for ds in DATASETS:
        n = sum(1 for r in rows if r["Dataset"] == ds)
        if n not in (25, 50, 300, 280):
            anomalies.append(f"{ds} N={n}")
    # 唯一键: (Image_ID, Dataset) 联合。M3FD 与 FMB 存在 28 个跨数据集重名 ID
    # (如 00210/00353 等, 两数据集命名空间冲突), 属数据集真实情况, 非评估错误。
    keys = [(r["Image_ID"], r["Dataset"]) for r in rows]
    if len(set(keys)) != len(keys):
        anomalies.append("duplicate (Image_ID, Dataset) — need investigate")
    for r in rows:
        for mk in METRICS:
            v = float(r[mk])
            if not np.isfinite(v):
                anomalies.append(f"non-finite {mk} at {r['Image_ID']}/{r['Dataset']}")
    for ds in DATASETS:
        recs = [r for r in rows if r["Dataset"] == ds]
        for mk in METRICS:
            vals = [float(r[mk]) for r in recs]
            s = stats(vals)
            if not (s["min"] <= s["mean"] <= s["max"]) or s["std"] < 0:
                anomalies.append(f"{ds}/{mk}: min<=mean<=max violated")
    # 已知说明: IVIF 的 SSIM = SSIM(F,IR)+SSIM(F,VIS), 两源图相似度高时 >1 属正常
    # (与论文历史数据一致, e.g. FMB=1.47)。此处仅记录, 不视为异常。
    if anomalies:
        with open(os.path.join(out_dir, "anomaly_report.txt"), 'w') as f:
            f.write("\n".join(anomalies))
        print(f"⚠️ anomaly_report.txt: {len(anomalies)} 项")
    else:
        print("✅ sanity check 全部通过（N=655, 无NaN/Inf, (Image_ID,Dataset) 唯一）")
    # 记录已知数据特性（供审计查阅）
    with open(os.path.join(out_dir, "known_notes.txt"), 'w') as f:
        f.write("已知数据特性（非异常）:\n")
        f.write("- IVIF SSIM = SSIM(F,IR) + SSIM(F,VIS), 两源图均高时可达 1.5-1.8, 属 Evaluator 口径正常值\n")
        f.write("- M3FD 与 FMB 存在 28 个跨数据集重名 Image_ID (如 00210, 00353), 数据集命名空间冲突, "
                "唯一键以 (Image_ID, Dataset) 联合为准\n")
    print("✅ known_notes.txt 已记录")

    print(f"\n🎉 {mkey} 全部完成 → {out_dir}")


if __name__ == '__main__':
    main()
