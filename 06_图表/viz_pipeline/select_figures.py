# -*- coding: utf-8 -*-
"""
===========================================================================
select_figures.py — 可视化候选图自动筛选（Deliverable 1）
===========================================================================
输入: 各方法单图指标 CSV（per_image_eval.py 生成，含 Image_ID + 指标列）
输出: 1) 综合增益 Top-15 排名
      2) 四类论文典型场景的 Top-10 候选清单

指标权重（按任务要求）:
  - 高频与边缘细节（高权重）: AG(x2.0), SF(x1.5), Qabf(x1.5)
  - 信息丰富度:               EN(x1.0), SCD(x1.0), MI(x1.0)
  - 视觉感知:                 VIF(x1.0)
  - 惩罚项: SSIM 相对跌幅超过全体平均劣势 1.5 倍 → 增益打 6 折

用法:
  python experiments/viz_pipeline/select_figures.py \
      --ours experiments/viz_pipeline/perimg_ours.csv \
      --baseline experiments/viz_pipeline/perimg_cddfuse.csv \
      --baseline-name CDDFuse
===========================================================================
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

WEIGHTS = {
    "AG": 2.0,    # 平均梯度 —— 边缘锐度
    "SF": 1.5,    # 空间频率 —— 纹理清晰度
    "Qabf": 1.5,  # 边缘保留度
    "EN": 1.0,    # 信息熵
    "SCD": 1.0,   # 差异相关和
    "MI": 1.0,    # 互信息
    "VIF": 1.0,   # 视觉保真度
    "SD": 0.5,    # 对比度
    "SSIM": 0.5,  # 结构相似度
    "CC": 0.5,    # 相关系数
}

SCENARIOS = {
    "场景1_强热目标与轮廓边缘": {
        "desc": "夜间/行人/车辆等强热目标场景",
        "keys": ["MI", "SF", "SD"],
        "why": "热目标由 IR 强度对比度主导 → MI/SF 增益最能体现 Ours 的全局语义对齐优势",
    },
    "场景2_丰富高频微小纹理": {
        "desc": "树叶/文字/砖纹等微纹理场景",
        "keys": ["AG", "Qabf", "SF"],
        "why": "微纹理由高频分量主导 → AG/Qabf 增益体现 DWConv 局部先验与 SG 提纯能力",
    },
    "场景3_极端光照与动态范围": {
        "desc": "逆光/过曝/暗部场景",
        "keys": ["EN", "SD", "SCD"],
        "why": "极端光照下信息保真度与对比度是核心 → EN/SD 增益体现选择性记忆的优势",
    },
    "场景4_自然度与无伪影过渡": {
        "desc": "整体自然、过渡平滑、无伪影",
        "keys": ["SSIM", "VIF", "CC"],
        "why": "感知自然度 → SSIM/VIF 持平或微优 + 无大幅惩罚即入选",
    },
}


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_id = {r["Image_ID"]: r for r in rows}
    return rows, by_id


def pct_delta(ours, base):
    """相对百分比增益，基线接近 0 时退化为绝对差"""
    try:
        o, b = float(ours), float(base)
    except (TypeError, ValueError):
        return 0.0
    if abs(b) < 1e-6:
        return 0.0
    return (o - b) / b * 100.0


def build_std_scaler(rows_by_id, ids):
    """按数据集计算基线各指标的标准差，用于标准化增益（防除零伪影）"""
    scaler = {}
    for ds in set(r["Dataset"] for r in rows_by_id.values()):
        ds_ids = [i for i in ids if rows_by_id[i]["Dataset"] == ds]
        for k in WEIGHTS:
            vals = [float(rows_by_id[i][k]) for i in ds_ids if k in rows_by_id[i]]
            std = float(np.std(vals)) if len(vals) > 1 else 1.0
            scaler[(ds, k)] = max(std, 1e-6)
    return scaler


def std_delta(row_o, row_b, scaler, k):
    """标准化增益: (ours-base)/std_base —— 以基线波动为单位，对近零基线稳健"""
    try:
        o, b = float(row_o[k]), float(row_b[k])
    except (TypeError, ValueError):
        return 0.0
    ds = row_o["Dataset"]
    return (o - b) / scaler.get((ds, k), 1e-6)


def composite_gain(row_o, row_b, scaler, ssim_penalty_threshold):
    """计算单图综合加权增益（标准化 delta + SSIM 惩罚）"""
    gain = 0.0
    for k, w in WEIGHTS.items():
        if k not in row_o or k not in row_b:
            continue
        gain += w * std_delta(row_o, row_b, scaler, k)
    # 惩罚项: SSIM 标准化跌幅超过 -1.0 个基线标准差 → 打 6 折
    ssim_d = std_delta(row_o, row_b, scaler, "SSIM")
    if ssim_d < ssim_penalty_threshold:
        gain *= 0.6
    return gain


def topk_by_keys(ranked, keys, scaler, k=10):
    """按场景指定指标的标准化加权增益重新排序取 Top-K"""
    scores = []
    for img_id, row_o, row_b, comp in ranked:
        s = sum(WEIGHTS.get(kk, 1.0) * std_delta(row_o, row_b, scaler, kk)
                for kk in keys if kk in row_o and kk in row_b)
        scores.append((img_id, row_o, row_b, s))
    scores.sort(key=lambda x: -x[3])
    return scores[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--baseline-name', default='Baseline')
    ap.add_argument('--top', type=int, default=15)
    args = ap.parse_args()

    rows_o, by_id_o = load_csv(args.ours)
    rows_b, by_id_b = load_csv(args.baseline)

    # 数据集分组
    by_ds = defaultdict(list)
    for img_id in by_id_o:
        by_ds[by_id_o[img_id]["Dataset"]].append(img_id)

    # 标准化缩放器（按数据集 × 指标计算基线标准差）
    scaler = build_std_scaler(by_id_b, list(by_id_b.keys()))
    ssim_penalty_threshold = -1.0  # SSIM 跌幅超过 1 个基线标准差 → 打折

    print(f"Ours 图像数: {len(rows_o)} | {args.baseline_name} 图像数: {len(rows_b)}")
    print("增益单位: 基线标准差数（σ）——对近零基线稳健，无除零伪影\n")

    # ── 1. 综合增益 Top-K ──
    ranked = []
    for img_id in by_id_o:
        if img_id not in by_id_b:
            continue
        g = composite_gain(by_id_o[img_id], by_id_b[img_id], scaler, ssim_penalty_threshold)
        ranked.append((img_id, by_id_o[img_id], by_id_b[img_id], g))
    ranked.sort(key=lambda x: -x[3])

    print("=" * 100)
    print(f"综合增益 Top-{args.top}（Ours vs {args.baseline_name}，标准化增益 + SSIM 惩罚）")
    print("=" * 100)
    print(f"{'排名':<4} {'Image_ID':<12} {'数据集':<10} {'综合增益σ':>10} "
          f"{'AGΔσ':>7} {'SFΔσ':>7} {'QabfΔσ':>7} {'MIΔσ':>7} {'ENΔσ':>7} {'SSIMΔσ':>8}")
    for rank, (img_id, row_o, row_b, g) in enumerate(ranked[:args.top], 1):
        print(f"{rank:<4} {img_id:<12} {row_o['Dataset']:<10} {g:>10.2f} "
              f"{std_delta(row_o, row_b, scaler, 'AG'):>7.2f} "
              f"{std_delta(row_o, row_b, scaler, 'SF'):>7.2f} "
              f"{std_delta(row_o, row_b, scaler, 'Qabf'):>7.2f} "
              f"{std_delta(row_o, row_b, scaler, 'MI'):>7.2f} "
              f"{std_delta(row_o, row_b, scaler, 'EN'):>7.2f} "
              f"{std_delta(row_o, row_b, scaler, 'SSIM'):>8.2f}")

    # ── 2. 场景化 Top-K ──
    print("\n" + "=" * 100)
    print("四类论文典型场景候选清单（标准化增益，单位 σ）")
    print("=" * 100)
    for scen, cfg in SCENARIOS.items():
        print(f"\n【{scen}】{cfg['desc']}")
        print(f"  排序依据: {' / '.join(cfg['keys'])} — {cfg['why']}")
        print(f"  {'Image_ID':<12} {'数据集':<10} {'综合增益σ':>10} " +
              " ".join(f"{k+'Δσ':>8}" for k in cfg['keys']))
        top = topk_by_keys(ranked, cfg['keys'], scaler, k=10)
        for img_id, row_o, row_b, s in top:
            print(f"  {img_id:<12} {row_o['Dataset']:<10} {s:>10.2f} " +
                  " ".join(f"{std_delta(row_o, row_b, scaler, k):>8.2f}" for k in cfg['keys']))


if __name__ == '__main__':
    main()
