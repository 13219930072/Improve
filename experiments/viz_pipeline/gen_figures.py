# -*- coding: utf-8 -*-
"""
===========================================================================
gen_figures.py — 论文对比图生成
===========================================================================
对选定图像生成 IR | VIS | CDDFuse | LAMF 四联对比图（含红绿框标注建议）。

用法:
  # 1) 生成两方法的融合 PNG
  python experiments/viz_pipeline/gen_figures.py --phase fuse \
      --pairs "TNO:06" "TNO:24" "RoadScene:FLIR_06392" "RoadScene:FLIR_07209"

  # 2) 拼四联图 + 自动标注 zoom-in 框
  python experiments/viz_pipeline/gen_figures.py --phase compose \
      --pairs "TNO:06" "TNO:24" "RoadScene:FLIR_06392" "RoadScene:FLIR_07209"

输出:
  experiments/viz_pipeline/figures/<ds>_<id>_quad.png   (四联图)
  experiments/viz_pipeline/figures/<ds>_<id>_<ours|cddfuse>.png (融合图)
===========================================================================
"""
import os
import sys
import argparse

import numpy as np
import torch

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_ROOT = "/home/hyhz/HT/test_img"
OURS_CKPT = "/home/hyhz/HT/code/MY-CDDFuse/experiments/LAMF_Full/checkpoints/LAMF_Full_Final_07-17-13-19.pth"
CDDFUSE_CKPT = "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main/models/CDDFuse_IVF.pth"
CDDFUSE_DIR = "/home/hyhz/HT/code/MY-CDDFuse/experiments/MMIF-CDDFuse-main"
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def strip(sd):
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in sd.items()}


def load_ours():
    from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction
    from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
    ckpt = torch.load(OURS_CKPT, map_location=DEVICE)
    E = Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = LocalEnhancedVSSBlock(dim=64).to(DEVICE); BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = DetailFeatureExtraction(num_layers=1).to(DEVICE); DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def load_cddfuse():
    sys.path.insert(0, CDDFUSE_DIR)
    import net as cdd_net
    ckpt = torch.load(CDDFUSE_CKPT, map_location=DEVICE)
    E = cdd_net.Restormer_Encoder().to(DEVICE); E.load_state_dict(strip(ckpt['DIDF_Encoder'])); E.eval()
    D = cdd_net.Restormer_Decoder().to(DEVICE); D.load_state_dict(strip(ckpt['DIDF_Decoder'])); D.eval()
    BF = cdd_net.BaseFeatureExtraction(dim=64, num_heads=8).to(DEVICE)
    BF.load_state_dict(strip(ckpt['BaseFuseLayer'])); BF.eval()
    DF = cdd_net.DetailFeatureExtraction(num_layers=1).to(DEVICE)
    DF.load_state_dict(strip(ckpt['DetailFuseLayer'])); DF.eval()
    return E, D, BF, DF


def fuse(E, D, BF, DF, ir, vis):
    with torch.no_grad():
        fV_B, fV_D, _ = E(vis)
        fI_B, fI_D, _ = E(ir)
        fF_B = BF(fI_B + fV_B)
        fF_D = DF(fI_D + fV_D)
        fused, _ = D(vis, fF_B, fF_D)
    return fused


def find_img(ds, sub, img_id):
    """在 ir/vi 目录中查找 img_id 的实际文件（支持任意扩展名）"""
    d = os.path.join(TEST_ROOT, ds, sub)
    for fn in os.listdir(d):
        if os.path.splitext(fn)[0] == img_id:
            return os.path.join(d, fn)
    raise FileNotFoundError(f"{d}/{img_id}.* 不存在")


def read_pair(ds, img_id):
    from utils.img_read_save import image_read_cv2
    ir_raw = image_read_cv2(find_img(ds, "ir", img_id), 'GRAY')
    vi_raw = image_read_cv2(find_img(ds, "vi", img_id), 'GRAY')
    ir = torch.FloatTensor(ir_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
    vi = torch.FloatTensor(vi_raw[np.newaxis, np.newaxis, ...] / 255.0).to(DEVICE)
    return ir, vi


def save_fused(fused, ds, img_id, tag):
    from utils.img_read_save import img_save
    os.makedirs(FIG_DIR, exist_ok=True)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    fi_np = np.squeeze((fused * 255.0).cpu().numpy()).astype(np.uint8)
    img_save(fi_np, f"{ds}_{img_id}_{tag}", FIG_DIR)
    return fi_np


def phase_fuse(pairs):
    print("[1/2] 生成融合图 ...")
    Eo, Do, BFo, DFo = load_ours()
    print("  ✅ LAMF 加载完成")
    # CDDFuse 需要独立进程（net 模块名冲突），此处仅对 ours 融合
    for ds, img_id in pairs:
        ir, vi = read_pair(ds, img_id)
        fused = fuse(Eo, Do, BFo, DFo, ir, vi)
        save_fused(fused, ds, img_id, "ours")
        print(f"  ✅ {ds}/{img_id} (LAMF)")
    # 注意: cddfuse 融合请单独运行 --method cddfuse


def phase_compose(pairs, zoom_regions):
    """拼四联图: IR | VIS | CDDFuse | LAMF"""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    for ds, img_id in pairs:
        from utils.img_read_save import image_read_cv2
        ir = image_read_cv2(find_img(ds, "ir", img_id), 'GRAY')
        vi = image_read_cv2(find_img(ds, "vi", img_id), 'GRAY')
        f_ours = image_read_cv2(os.path.join(FIG_DIR, f"{ds}_{img_id}_ours.png"), 'GRAY')
        f_cdd = image_read_cv2(os.path.join(FIG_DIR, f"{ds}_{img_id}_cddfuse.png"), 'GRAY')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Infrared", "Visible", "CDDFuse", "LAMF (Ours)"]
        imgs = [ir, vi, f_cdd, f_ours]
        for ax, img, t in zip(axes, imgs, titles):
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(t, fontsize=14)
            ax.axis('off')

        # zoom 框
        for color, (region, label) in zoom_regions.get((ds, img_id), {}).items():
            for ax in axes[1:]:  # VIS 起
                y0, x0, h, w = region
                rect = plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=3)
                ax.add_patch(rect)
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{ds}_{img_id}_quad.png")
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['fuse', 'compose'], required=True)
    ap.add_argument('--method', choices=['ours', 'cddfuse'], default='ours')
    ap.add_argument('--pairs', nargs='+', required=True,
                    help='图像对列表: "TNO:06" "RoadScene:FLIR_06392"')
    ap.add_argument('--zoom', default='', help='zoom 区域 JSON 文件路径（可选）')
    args = ap.parse_args()

    pairs = []
    for p in args.pairs:
        ds, img_id = p.split(':', 1)
        pairs.append((ds, img_id))

    if args.phase == 'fuse':
        if args.method == 'ours':
            E, D, BF, DF = load_ours()
        else:
            E, D, BF, DF = load_cddfuse()
        print(f"✅ {args.method} 加载完成")
        for ds, img_id in pairs:
            ir, vi = read_pair(ds, img_id)
            fused = fuse(E, D, BF, DF, ir, vi)
            save_fused(fused, ds, img_id, args.method)
            print(f"  ✅ {ds}/{img_id} ({args.method})")
    else:
        zoom_regions = {}
        if args.zoom and os.path.exists(args.zoom):
            import json
            zoom_regions = json.load(open(args.zoom))
        phase_compose(pairs, zoom_regions)


if __name__ == '__main__':
    main()
