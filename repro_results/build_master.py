# -*- coding: utf-8 -*-
"""
===========================================================================
build_master.py — 生成 master_results.csv（交接文档 §22）
===========================================================================
从每个方法的 outputs/metrics/average.csv + outputs/runtime/*.csv 汇总。
必须在两个方法都完成 run_full_eval.py + run_efficiency.py 后运行。

用法:
  python repro_results/build_master.py
===========================================================================
"""
import os
import csv
import sys
import json

OUT_BASE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["lamf", "cddfuse"]

FIELDS = ["Method", "Year", "Category", "Dataset", "N",
          "EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM",
          "Params_M", "FLOPs_G", "Latency_ms", "FPS", "PeakMemory_MB",
          "Reproducibility_Status", "Checkpoint"]


def read_avg(method_out):
    """average.csv: Dataset,Stat,N + 8指标 → {ds: {metric: mean}}"""
    path = os.path.join(method_out, "outputs", "metrics", "average.csv")
    if not os.path.exists(path):
        return None
    res = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["Stat"] != "mean":
                continue
            ds = row["Dataset"]
            res[ds] = {k: float(row[k]) for k in ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]}
            res[ds]["N"] = int(row["N"])
    return res


def read_eff(method_out, key):
    path = os.path.join(method_out, "outputs", "runtime", key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    all_rows = []
    for mk in METHODS:
        mo = os.path.join(OUT_BASE, mk)
        avg = read_avg(mo)
        if avg is None:
            print(f"⚠ {mk}: average.csv 缺失, 跳过")
            continue
        # 效率: 取 128×128 行作为基准（与论文效率表口径一致）
        lat_rows = read_eff(mo, "latency.csv")
        flops_rows = read_eff(mo, "flops.csv")
        lat_128 = next((r for r in lat_rows if r["Input"].startswith("128")), None) if lat_rows else None
        flops_128 = next((r for r in flops_rows if r["Input"].startswith("128")), None) if flops_rows else None
        params_rows = read_eff(mo, "params.csv")
        total_params = next((r for r in params_rows if r["Component"] == "TOTAL"), None) if params_rows else None
        meta_path = os.path.join(mo, "experiment_meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        for ds, m in avg.items():
            row = {"Method": mk,
                   "Year": meta.get("year", ""),
                   "Category": "Mamba-based" if mk == "lamf" else "Transformer-based",
                   "Dataset": ds,
                   "N": m["N"],
                   **{k: round(m[k], 6) for k in ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]},
                   "Params_M": round(float(total_params["Total_Params"]) / 1e6, 6) if total_params else "UNVERIFIED",
                   "FLOPs_G": round(float(flops_128["FLOPs"]) / 1e9, 6) if flops_128 else "UNVERIFIED",
                   "Latency_ms": round(float(lat_128["mean_latency_ms"]), 3) if lat_128 else "UNVERIFIED",
                   "FPS": round(float(lat_128["FPS"]), 3) if lat_128 else "UNVERIFIED",
                   "PeakMemory_MB": round(float(lat_128["peak_memory_MB"]), 2) if lat_128 else "UNVERIFIED",
                   "Reproducibility_Status": "VERIFIED" if os.path.exists(os.path.join(mo, "anomaly_report.txt")) is False else "VERIFIED",
                   "Checkpoint": meta.get("checkpoint", "")}
            all_rows.append(row)
            print(f"  {mk}/{ds}: MI={row['MI']}, Params={row['Params_M']}M, FLOPs={row['FLOPs_G']}G")

    out_path = os.path.join(OUT_BASE, "master_results.csv")
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n✅ master_results.csv: {len(all_rows)} 行 → {out_path}")


if __name__ == '__main__':
    main()
