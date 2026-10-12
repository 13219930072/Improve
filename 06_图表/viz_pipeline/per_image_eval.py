# -*- coding: utf-8 -*-
"""
===========================================================================
per_image_eval.py — 单图指标评估器
===========================================================================
对测试集逐图计算客观指标，输出 CSV（每行一张图），供可视化候选图筛选。

用法:
  # 评估 LAMF (Ours)
  python experiments/viz_pipeline/per_image_eval.py --method ours \
      --datasets TNO RoadScene --out experiments/viz_pipeline/perimg_ours.csv

  # 评估 CDDFuse (Baseline) —— 独立进程运行，避免 net.py 模块名冲突
  python experiments/viz_pipeline/per_image_eval.py --method cddfuse \
      --datasets TNO RoadScene --out experiments/viz_pipeline/perimg_cddfuse.csv

输出 CSV 列:
  Image_ID, Dataset, EN, SD, SF, AG, MI, SCD, VIF, Qabf, SSIM, CC, PSNR, MSE
===========================================================================
"""
import os
import sys
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_ROOT = "/home/hyhz/HT/test_img"

# LAMF 官方权重（实验 E，July 17 正确版 —— 与论文数据 eval_0717-1949 一致）
OURS_CKPT = "/home/hyhz/HT/code/MY-CDDFuse/02_消融实验/E_LAMF完整/checkpoints/LAMF_Full_Final_07-17-13-19.pth"
# CDDFuse 官方权重（含 1 层 INN 的版本，与 test_IVF.py 一致）
CDDFUSE_CKPT = "/home/hyhz/HT/code/MY-CDDFuse/04_复现对比/CDDFuse官方代码/models/CDDFuse_IVF.pth"
CDDFUSE_DIR = "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main"

METRIC_FIELDS = ["EN", "SD", "SF", "AG", "MI", "SCD", "VIF", "Qabf", "SSIM", "CC", "PSNR", "MSE"]


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def load_ours():
    """加载 LAMF (E): LESS-Mamba + NAFDetailBlock"""
    from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
    from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
    ckpt = torch.load(OURS_CKPT, map_location=DEVICE)
    E = Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = LocalEnhancedVSSBlock(dim=64).to(DEVICE); BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = DetailFeatureExtraction(num_layers=1).to(DEVICE); DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def load_cddfuse():
    """加载 CDDFuse 官方权重（原版 INN Detail 分支）"""
    sys.path.insert(0, CDDFUSE_DIR)
    import net as cdd_net  # CDDFuse 自己的 net.py
    ckpt = torch.load(CDDFUSE_CKPT, map_location=DEVICE)
    E = cdd_net.Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = cdd_net.Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = cdd_net.BaseFeatureExtraction(dim=64, num_heads=8).to(DEVICE)
    BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = cdd_net.DetailFeatureExtraction(num_layers=1).to(DEVICE)
    DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def fuse_one(E, D, BF, DF, ir, vis):
    with torch.no_grad():
        fV_B, fV_D, _ = E(vis)
        fI_B, fI_D, _ = E(ir)
        fF_B = BF(fI_B + fV_B)
        fF_D = DF(fI_D + fV_D)
        fused, _ = D(vis, fF_B, fF_D)
    return fused


# 与 eval_all.py 完全一致的口径:
#   融合输出 min-max 归一化 → *255 → uint8 → img_save 存 PNG
#   → image_read_cv2 读回 (float32, np.round) → 以 float32 计算指标
_FUSED_TMP = "/tmp/viz_fused_tmp"


def to_eval_float(fused, img_name):
    from utils.img_read_save import img_save, image_read_cv2
    import uuid
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    fi_np = np.squeeze((fused * 255.0).cpu().numpy()).astype(np.uint8)
    os.makedirs(_FUSED_TMP, exist_ok=True)
    tag = f"{img_name}_{uuid.uuid4().hex[:6]}"
    img_save(fi_np, tag, _FUSED_TMP)
    return image_read_cv2(os.path.join(_FUSED_TMP, tag + ".png"), 'GRAY')  # float32


def compute_metrics(fused_np, ir_np, vis_np):
    from utils.Evaluator import Evaluator
    return {
        "EN": Evaluator.EN(fused_np),
        "SD": Evaluator.SD(fused_np),
        "SF": Evaluator.SF(fused_np),
        "AG": Evaluator.AG(fused_np),
        "MI": Evaluator.MI(fused_np, ir_np, vis_np),
        "SCD": Evaluator.SCD(fused_np, ir_np, vis_np),
        "VIF": Evaluator.VIFF(fused_np, ir_np, vis_np),
        "Qabf": Evaluator.Qabf(fused_np, ir_np, vis_np),
        "SSIM": Evaluator.SSIM(fused_np, ir_np, vis_np),
        "CC": Evaluator.CC(fused_np, ir_np, vis_np),
        "PSNR": Evaluator.PSNR(fused_np, ir_np, vis_np),
        "MSE": Evaluator.MSE(fused_np, ir_np, vis_np),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ours', 'cddfuse'], required=True)
    parser.add_argument('--datasets', nargs='+', default=['TNO', 'RoadScene'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    if args.method == 'ours':
        E, D, BF, DF = load_ours()
        print("[ours] LAMF (E) 加载完成")
    else:
        E, D, BF, DF = load_cddfuse()
        print("[cddfuse] CDDFuse 官方权重加载完成")

    from utils.img_read_save import image_read_cv2, img_save

    rows = []
    t0 = time.time()
    for ds in args.datasets:
        ir_dir = os.path.join(TEST_ROOT, ds, "ir")
        vi_dir = os.path.join(TEST_ROOT, ds, "vi")
        if not os.path.isdir(ir_dir) or not os.path.isdir(vi_dir):
            print(f"  ⚠ 跳过 {ds}（目录不存在）")
            continue
        ir_files = sorted(os.listdir(ir_dir))
        print(f"▶ {ds}: {len(ir_files)} 对图像")
        for i, fn in enumerate(ir_files):
            ir_raw = image_read_cv2(os.path.join(ir_dir, fn), 'GRAY')   # float32
            vi_raw = image_read_cv2(os.path.join(vi_dir, fn), 'GRAY')   # float32
            ir = torch.FloatTensor(ir_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
            vis = torch.FloatTensor(vi_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
            fused = fuse_one(E, D, BF, DF, ir, vis)
            # 复刻 eval_all.py 口径: 保存 PNG 后读回 float32 再算指标
            fused_f = to_eval_float(fused, os.path.splitext(fn)[0])
            m = compute_metrics(fused_f, ir_raw, vi_raw)
            row = {"Image_ID": os.path.splitext(fn)[0], "Dataset": ds}
            row.update(m)
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(ir_files)} ({(time.time()-t0)/(i+1):.2f}s/图)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Image_ID", "Dataset"] + METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 完成 {len(rows)} 行 → {args.out}")


if __name__ == '__main__':
    main()
