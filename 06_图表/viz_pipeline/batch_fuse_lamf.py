# -*- coding: utf-8 -*-
"""
===========================================================================
batch_fuse_lamf.py — LAMF 全量融合结果生成（完整网络结构 · 实验 E）
===========================================================================
对 TNO / RoadScene / M3FD / FMB 全部图像对生成融合结果 PNG 并保留。

与论文评估管线口径一致:
  网络输入 /255.0 → 融合 → min-max 归一化 → ×255 → uint8 → 保存 PNG

用法:
  python experiments/viz_pipeline/batch_fuse_lamf.py

输出:
  experiments/viz_pipeline/fused_LAMF/<数据集>/<Image_ID>.png
===========================================================================
"""
import os
import sys
import time

import numpy as np
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_ROOT = "/home/hyhz/HT/test_img"
# 正确权重: 与论文数据 eval_0717-1949 一致的 July 17 版
OURS_CKPT = "/home/hyhz/HT/code/MY-CDDFuse/02_消融实验/E_LAMF完整/checkpoints/LAMF_Full_Final_07-17-13-19.pth"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_LAMF")

DATASETS = ["TNO", "RoadScene", "M3FD", "FMB"]


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def load_lamf():
    from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
    from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
    ckpt = torch.load(OURS_CKPT, map_location=DEVICE)
    E = Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = LocalEnhancedVSSBlock(dim=64).to(DEVICE); BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = DetailFeatureExtraction(num_layers=1).to(DEVICE); DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def fuse_one(E, D, BF, DF, ir, vis):
    with torch.no_grad():
        fV_B, fV_D, _ = E(vis)
        fI_B, fI_D, _ = E(ir)
        fF_B = BF(fI_B + fV_B)
        fF_D = DF(fI_D + fV_D)
        fused, _ = D(vis, fF_B, fF_D)
    return fused


def find_img(ds, sub, img_id):
    d = os.path.join(TEST_ROOT, ds, sub)
    for fn in os.listdir(d):
        if os.path.splitext(fn)[0] == img_id:
            return os.path.join(d, fn)
    raise FileNotFoundError(f"{d}/{img_id}.* 不存在")


def main():
    from utils.img_read_save import image_read_cv2, img_save

    E, D, BF, DF = load_lamf()
    print(f"✅ LAMF 加载完成 (权重: {os.path.basename(OURS_CKPT)})")
    print(f"输出: {OUT_DIR}\n")

    total = 0
    t0 = time.time()
    for ds in DATASETS:
        ir_dir = os.path.join(TEST_ROOT, ds, "ir")
        if not os.path.isdir(ir_dir):
            print(f"  ⚠ 跳过 {ds}（目录不存在）")
            continue
        ids = sorted(os.path.splitext(fn)[0] for fn in os.listdir(ir_dir))
        out_ds = os.path.join(OUT_DIR, ds)
        os.makedirs(out_ds, exist_ok=True)
        print(f"▶ {ds}: {len(ids)} 张")
        for i, img_id in enumerate(ids):
            ir_raw = image_read_cv2(find_img(ds, "ir", img_id), 'GRAY')
            vi_raw = image_read_cv2(find_img(ds, "vi", img_id), 'GRAY')
            ir = torch.FloatTensor(ir_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
            vis = torch.FloatTensor(vi_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
            fused = fuse_one(E, D, BF, DF, ir, vis)
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
            fi_np = np.squeeze((fused * 255.0).cpu().numpy()).astype(np.uint8)
            img_save(fi_np, img_id, out_ds)
            total += 1
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(ids)}")
        print(f"  ✅ {ds} 完成")

    elapsed = time.time() - t0
    print(f"\n🎉 全部完成: 共 {total} 张 → {OUT_DIR}")
    print(f"   耗时 {elapsed:.0f}s ({elapsed/max(total,1):.1f}s/张)")


if __name__ == '__main__':
    main()
