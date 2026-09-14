#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
LAMF-Fuse 全自动评估脚本 (Unified Evaluation Pipeline)
===============================================================================
功能一览:
  1. 一键评估 test_img/ 下所有数据集
  2. 支持同时评测多个模型的 .pth 权重文件
  3. 8项标准指标 (EN, SD, SF, MI, SCD, VIF, Qabf, SSIM) + 扩展指标 (AG, CC, PSNR)
  4. 效率统计 (推理时间, FPS, 参数量, FLOPs)
  5. 结果自动写入 Excel (每个模型一个 sheet)
  6. 评估完成后自动删除融合图像 (可选)
  7. 进度条 + 详细日志

使用方法:
  python eval_all.py

  ── 自定义模型 ──
  修改下方 CONFIG 区域中的 CKPT_LIST，每项填 (模型名, 权重路径)

  ── 命令行覆盖 ──
  python eval_all.py --ckpt models/xxx.pth --name MyModel
  python eval_all.py --keep      # 保留融合图像不删除
  python eval_all.py --no-flops  # 跳过 FLOPs 计算 (更快)
===============================================================================
"""

import os
import sys
import time
import shutil
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ============================================================
# 添加项目根目录到 path，确保在任何位置运行都能正确导入
# ============================================================
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'utils'))

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from utils.lamf_mamba_layer import OriginalVSSBlock, LocalEnhancedVSSBlock
from utils.Evaluator import Evaluator
from utils.img_read_save import img_save, image_read_cv2

# 实验公共模块（SimpleDetailFusion）
sys.path.insert(0, os.path.join(PROJ_ROOT, 'experiments'))
from common import SimpleDetailFusion

# ============================================================
# ======================  配 置 区 域  ========================
# ============================================================

# ── 测试数据根目录 ──
#  目录结构需为: test_img/<数据集名>/ir/*.png  +  test_img/<数据集名>/vi/*.png
TEST_ROOT = r"/home/hyhz/HT/test_img"

# ── 待评测模型列表 ──
#  格式: (模型显示名称, 权重文件路径)
#  可添加多个，每个会独立出一行/一列结果
CKPT_LIST = [
    ("LAMF_Full", r"/home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final.pth"),
    # ("CDDFuse (原版)",       r"/home/hyhz/HT/cddfuse_data/models/CDDFuse_IVF.pth"),
    # ("MyModel-V2",           r"models/xxx.pth"),
]

# ── 输出目录 ──
OUTPUT_DIR = "eval_results"               # Excel 存放目录
FUSED_IMG_DIR = "eval_results/fused_tmp"  # 融合图像暂存目录 (评估后删除)

# ── 默认消融变体 ──
#  用于 build_model() 选择 BaseFuseLayer / DetailFuseLayer 的类
#  A: Transformer + SimpleDetailFusion     (CDDFuse 基线)
#  C: Transformer + NAFDetailBlock SG      (+SimpleGate)
#  D: OriginalVSSBlock SiLU + SimpleDetail (+原生 MambaIR)
#  D+:OriginalVSSBlock SiLU + NAFDetail SG (桥接)
#  E: LocalEnhancedVSSBlock SG + NAFDetail (LAMF 最终)
DEFAULT_VARIANT = "E"

# ── 行为开关 ──
DELETE_AFTER_EVAL = True    # 评估完成后删除融合图像
COMPUTE_FLOPs = True         # 是否计算 FLOPs + 参数量 (需要 thop 库)
REDUCE_VIFF = False          # 图像太大导致 VIFF 太慢时，自动缩小到 max(H,W)=256
CALC_EXPANDED_METRICS = True # 额外计算 AG, CC, PSNR, MSE

# ── FLOPs 计算用的输入尺寸 ──
FLOPs_H, FLOPs_W = 128, 128

# ── 设备 ──
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# ====================  工 具 函 数  ==========================
# ============================================================

def find_all_datasets(test_root):
    """
    自动发现 test_root 下的所有数据集目录。
    规则：包含 ir/ 和 vi/ 子文件夹的目录即视为数据集。
    """
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"测试根目录不存在: {test_root}")

    datasets = []
    for name in sorted(os.listdir(test_root)):
        d = os.path.join(test_root, name)
        if os.path.isdir(d):
            ir_dir = os.path.join(d, "ir")
            vi_dir = os.path.join(d, "vi")
            if os.path.isdir(ir_dir) and os.path.isdir(vi_dir):
                datasets.append(name)
    if not datasets:
        raise RuntimeError(f"在 {test_root} 下未找到任何包含 ir/ 和 vi/ 的数据集")
    return datasets


def safe_remove_dir(path):
    """递归删除目录，无目录则静默跳过"""
    try:
        shutil.rmtree(path)
        print(f"  ✓ 已清理临时图像目录: {path}")
    except Exception as e:
        print(f"  ⚠ 清理失败 ({path}): {e}")


# ============================================================
# ====================  模 型 加 载  ==========================
# ============================================================

def build_model(variant="E"):
    """
    按消融变体构建四个子网络。

    variant 映射:
      A:  BaseFeatureExtraction       + SimpleDetailFusion
      C:  BaseFeatureExtraction       + DetailFeatureExtraction (NAFDetail SG)
      D:  OriginalVSSBlock (SiLU)     + SimpleDetailFusion
      D+: OriginalVSSBlock (SiLU)     + DetailFeatureExtraction (NAFDetail SG)
      E:  LocalEnhancedVSSBlock (SG)  + DetailFeatureExtraction (NAFDetail SG)
    """
    encoder = Restormer_Encoder().to(DEVICE)
    decoder = Restormer_Decoder().to(DEVICE)

    if variant in ('A', 'C'):
        base_fuse = BaseFeatureExtraction(dim=64, num_heads=8).to(DEVICE)
    elif variant in ('D', 'D+', 'D_plus'):
        base_fuse = OriginalVSSBlock(dim=64).to(DEVICE)
    elif variant == 'E':
        base_fuse = LocalEnhancedVSSBlock(dim=64).to(DEVICE)
    else:
        raise ValueError(f"未知 variant: {variant}，可选 A/C/D/D+/E")

    if variant in ('A', 'D'):
        detail_fuse = SimpleDetailFusion(dim=64).to(DEVICE)
    elif variant in ('C', 'D+', 'D_plus', 'E'):
        detail_fuse = DetailFeatureExtraction(num_layers=1, dim=64).to(DEVICE)
    else:
        raise ValueError(f"未知 variant: {variant}，可选 A/C/D/D+/E")

    return encoder, decoder, base_fuse, detail_fuse


def strip_module_prefix(state_dict):
    """移除 DataParallel 的 'module.' 前缀（如果有的话）。"""
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('module.', '', 1) if k.startswith('module.') else k] = v
    return new_sd


def load_checkpoint(ckpt_path, encoder, decoder, base_fuse, detail_fuse):
    """加载权重，自动兼容 DataParallel 包装的权重。"""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder.load_state_dict(strip_module_prefix(ckpt['DIDF_Encoder']))
    decoder.load_state_dict(strip_module_prefix(ckpt['DIDF_Decoder']))
    base_fuse.load_state_dict(strip_module_prefix(ckpt['BaseFuseLayer']))
    detail_fuse.load_state_dict(strip_module_prefix(ckpt['DetailFuseLayer']))
    print(f"  ✓ 权重加载成功: {ckpt_path}")


def count_params_and_flops(encoder, decoder, base_fuse, detail_fuse):
    """
    统计参数量 & FLOPs。
    FLOPs 以 (1, 1, H, W) 的 vis+ir 双输入计算完整前向。
    """
    H, W = FLOPs_H, FLOPs_W

    # 参数量
    total_params = 0
    for m in [encoder, decoder, base_fuse, detail_fuse]:
        total_params += sum(p.numel() for p in m.parameters())

    # FLOPs
    flops_str = "N/A"
    try:
        from thop import profile
        encoder.eval(); decoder.eval(); base_fuse.eval(); detail_fuse.eval()
        x_vis = torch.randn(1, 1, H, W).to(DEVICE)
        x_ir = torch.randn(1, 1, H, W).to(DEVICE)

        # 分别对 encoder 和 fusion 计算 FLOPs（因为 Decoder 输入动态尺寸）
        flops_e = profile(encoder, inputs=(x_vis,), verbose=False)[0]
        flops_d = profile(decoder, inputs=(torch.randn(1, 1, H, W).to(DEVICE),
                                            torch.randn(1, 64, H, W).to(DEVICE),
                                            torch.randn(1, 64, H, W).to(DEVICE)), verbose=False)[0]

        feat_v_b, feat_v_d, _ = encoder(x_vis)
        feat_i_b, feat_i_d, _ = encoder(x_ir)
        flops_bf = profile(base_fuse, inputs=(feat_v_b + feat_i_b,), verbose=False)[0]
        flops_df = profile(detail_fuse, inputs=(feat_v_d + feat_i_d,), verbose=False)[0]

        total_flops = (flops_e * 2 + flops_bf + flops_df + flops_d) / 1e9
        flops_str = f"{total_flops:.2f}"
    except ImportError:
        flops_str = "thop未安装"
    except Exception as e:
        flops_str = f"计算失败({e})"

    return total_params, flops_str


# ============================================================
# ====================  推 理 流 程  ==========================
# ============================================================

@torch.no_grad()
def fuse_one_dataset(encoder, decoder, base_fuse, detail_fuse,
                     test_folder, out_folder):
    """
    对一个数据集完成全部图像融合，同时统计时间。
    返回: (avg_time_s, avg_fps, img_count, avg_H, avg_W)
    """
    encoder.eval(); decoder.eval(); base_fuse.eval(); detail_fuse.eval()

    ir_dir = os.path.join(test_folder, "ir")
    vi_dir = os.path.join(test_folder, "vi")
    img_names = sorted(os.listdir(ir_dir))
    os.makedirs(out_folder, exist_ok=True)

    total_time = 0.0
    total_pixels = 0
    n = 0

    for img_name in tqdm(img_names, desc="  融合中", ncols=80):
        ir = image_read_cv2(os.path.join(ir_dir, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        vi = image_read_cv2(os.path.join(vi_dir, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0

        _, _, H, W = ir.shape
        total_pixels += H * W
        n += 1

        ir_t = torch.FloatTensor(ir).to(DEVICE)
        vi_t = torch.FloatTensor(vi).to(DEVICE)

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ----- 核心推理 -----
        fv_b, fv_d, _ = encoder(vi_t)
        fi_b, fi_d, _ = encoder(ir_t)
        ff_b = base_fuse(fv_b + fi_b)
        ff_d = detail_fuse(fv_d + fi_d)
        fused, _ = decoder(vi_t, ff_b, ff_d)
        # -------------------

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        fi_np = np.squeeze((fused * 255).cpu().numpy()).astype(np.uint8)
        img_save(fi_np, img_name.split('.')[0], out_folder)

    avg_t = total_time / n if n else 0
    fps = 1.0 / avg_t if avg_t > 0 else 0
    avg_px = total_pixels / n if n else 0
    avg_h = int(np.sqrt(avg_px))
    avg_w = int(avg_px / avg_h) if avg_h else 0
    return avg_t, fps, n, avg_h, avg_w


# ============================================================
# ====================  指 标 计 算  ==========================
# ============================================================

def compute_metrics_for_dataset(ir_dir, vi_dir, fused_dir, reduce_viff=False):
    """
    遍历融合结果，计算 8 项基础指标 + 可选扩展指标。
    返回: OrderedDict{指标名: 均值}
    """
    img_names = sorted(os.listdir(ir_dir))
    base_keys = ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]
    expanded_keys = ["AG", "CC", "PSNR", "MSE"]
    all_keys = base_keys + expanded_keys if CALC_EXPANDED_METRICS else base_keys

    accum = OrderedDict((k, 0.0) for k in all_keys)
    N = len(img_names)

    for img_name in img_names:
        name_no_ext = img_name.split('.')[0]
        ir = image_read_cv2(os.path.join(ir_dir, img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(vi_dir, img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(fused_dir, name_no_ext + ".png"), 'GRAY')

        # 针对极低分辨率图的 SSIM 保护 (win_size=7 -> 至少需要 7x7)
        try:
            accum["SSIM"] += Evaluator.SSIM(fi, ir, vi)
        except Exception as e:
            print(f"  ⚠ SSIM failed for {img_name}: {e}")
            accum["SSIM"] += 0.0

        # VIFF 对大图极慢，可选择降采样加速
        if reduce_viff and max(fi.shape) > 256:
            scale = 256.0 / max(fi.shape)
            new_h, new_w = int(fi.shape[0] * scale), int(fi.shape[1] * scale)
            from cv2 import resize
            fi_s = resize(fi, (new_w, new_h))
            ir_s = resize(ir, (new_w, new_h))
            vi_s = resize(vi, (new_w, new_h))
            accum["VIF"] += Evaluator.VIFF(fi_s, ir_s, vi_s)
        else:
            accum["VIF"] += Evaluator.VIFF(fi, ir, vi)

        # 其余指标
        accum["EN"]   += Evaluator.EN(fi)
        accum["SD"]   += Evaluator.SD(fi)
        accum["SF"]   += Evaluator.SF(fi)
        accum["MI"]   += Evaluator.MI(fi, ir, vi)
        accum["SCD"]  += Evaluator.SCD(fi, ir, vi)
        accum["Qabf"] += Evaluator.Qabf(fi, ir, vi)

        if CALC_EXPANDED_METRICS:
            accum["AG"]   += Evaluator.AG(fi)
            accum["CC"]   += Evaluator.CC(fi, ir, vi)
            accum["PSNR"] += Evaluator.PSNR(fi, ir, vi)
            accum["MSE"]  += Evaluator.MSE(fi, ir, vi)

    for k in accum:
        accum[k] /= N
    return accum


# ============================================================
# ====================  主 评 估 流 程  ========================
# ============================================================

def evaluate_all():
    """主函数：遍历模型 → 遍历数据集 → 融合 → 计算指标 → 写Excel → 清理"""
    # 命令行参数解析
    args = _parse_args()

    # 覆盖配置（如果命令行提供了参数）
    ckpt_list = CKPT_LIST
    if args.ckpt:
        name = args.name if args.name else os.path.splitext(os.path.basename(args.ckpt))[0]
        ckpt_list = [(name, args.ckpt)]
    delete_after = DELETE_AFTER_EVAL and not args.keep
    compute_flops = COMPUTE_FLOPs and not args.no_flops
    variant = args.variant or DEFAULT_VARIANT

    # 发现数据集
    datasets = find_all_datasets(args.test_root or TEST_ROOT)
    test_root = args.test_root or TEST_ROOT

    print("=" * 90)
    print("  LAMF-Fuse 全自动评估")
    print("=" * 90)
    print(f"  设备: {DEVICE}")
    print(f"  测试根目录: {test_root}")
    print(f"  发现数据集: {len(datasets)} 个 → {datasets}")
    print(f"  待评测模型: {len(ckpt_list)} 个")
    for name, path in ckpt_list:
        print(f"    - {name}: {path}")
    print(f"  消融变体: {variant} (Base/Detail 架构自动匹配)")
    print(f"  评估后删除融合图: {'是' if delete_after else '否'}")
    print("=" * 90)

    # ── 准备输出目录 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Excel writer ──
    timestamp = time.strftime("%m%d-%H%M")
    excel_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    all_models_summary = []  # 用于汇总 sheet

    # ── 逐模型评测 ──
    for model_idx, (model_name, ckpt_path) in enumerate(ckpt_list):
        print(f"\n{'─' * 80}")
        print(f"[{model_idx+1}/{len(ckpt_list)}] 模型: {model_name}")
        print(f"  权重: {ckpt_path}")

        # 1. 构建并加载模型
        encoder, decoder, base_fuse, detail_fuse = build_model(variant)
        load_checkpoint(ckpt_path, encoder, decoder, base_fuse, detail_fuse)

        # 2. 统计参数量 & FLOPs
        if compute_flops:
            print("  正在计算参数量 & FLOPs ...")
            params, flops = count_params_and_flops(encoder, decoder, base_fuse, detail_fuse)
            print(f"  Params: {params/1e6:.2f} M  |  FLOPs: {flops} G")
        else:
            params, flops = 0, "N/A"

        # 3. 逐数据集评测
        model_results = []  # list of OrderedDict

        for ds_name in datasets:
            print(f"\n  ── 数据集: {ds_name} ──")
            ir_dir = os.path.join(test_root, ds_name, "ir")
            vi_dir = os.path.join(test_root, ds_name, "vi")
            fused_dir = os.path.join(FUSED_IMG_DIR, model_name, ds_name)

            # 3a. 融合
            t_avg, fps, n_img, avg_h, avg_w = fuse_one_dataset(
                encoder, decoder, base_fuse, detail_fuse,
                test_folder=os.path.join(test_root, ds_name),
                out_folder=fused_dir
            )

            # 3b. 计算质量指标
            print("  计算质量指标...")
            metrics = compute_metrics_for_dataset(ir_dir, vi_dir, fused_dir,
                                                  reduce_viff=REDUCE_VIFF)

            # 汇总一行
            row = OrderedDict()
            row["Model"] = model_name
            row["Dataset"] = ds_name
            row["#Images"] = n_img
            row["AvgRes"] = f"{avg_h}×{avg_w}"
            row.update(metrics)
            row["Time(s)"] = round(t_avg, 4)
            row["FPS"] = round(fps, 2)
            row["Params(M)"] = round(params / 1e6, 2) if isinstance(params, (int, float)) else params
            row["FLOPs(G)"] = flops
            model_results.append(row)

            # 3c. 按需清理融合图像
            if delete_after:
                safe_remove_dir(fused_dir)

        # 4. 将此模型结果写入一个 sheet
        df_model = pd.DataFrame(model_results)
        sheet_name = model_name[:31]  # Excel sheet 名最长 31 字符
        df_model.to_excel(writer, sheet_name=sheet_name, index=False)

        # 收集到汇总表
        all_models_summary.extend(model_results)

    # ── 汇总 sheet ──
    df_all = pd.DataFrame(all_models_summary)
    df_all.to_excel(writer, sheet_name="Summary", index=False)

    # ── 写入并关闭 ──
    writer.close()

    # ── 清理顶层临时目录 ──
    top_tmp = os.path.join(FUSED_IMG_DIR)
    if delete_after and os.path.isdir(top_tmp):
        safe_remove_dir(top_tmp)

    # ── 在控制台打印汇总表 ──
    _print_summary(df_all)

    print(f"\n{'=' * 90}")
    print(f"  全部完成！结果已保存至: {os.path.abspath(excel_path)}")
    print(f"{'=' * 90}\n")


def _print_summary(df):
    """美化的控制台汇总打印。"""
    if df.empty:
        return

    # 指标列（数值型）
    metric_cols = ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM",
                   "AG", "CC", "PSNR", "MSE", "Time(s)", "FPS"]

    print(f"\n{'=' * 130}")
    print("  评 估 汇 总")
    print(f"{'=' * 130}")

    # 表头
    header_cols = ["Model", "Dataset"] + [c for c in metric_cols if c in df.columns]
    header = f"{'Model':<28s} {'Dataset':<12s}"
    for c in metric_cols:
        if c in df.columns:
            header += f" {c:>8s}"
    print(header)
    print("-" * 130)

    # 逐行
    for _, row in df.iterrows():
        line = f"{str(row['Model']):<28s} {str(row['Dataset']):<12s}"
        for c in metric_cols:
            if c in df.columns:
                val = row[c]
                if isinstance(val, float):
                    line += f" {val:>8.2f}" if c not in ("Time(s)", "PSNR") else f" {val:>8.3f}"
                else:
                    line += f" {str(val):>8s}"
        print(line)

    # 按模型汇总（平均）
    print(f"\n{'─' * 130}")
    print("  各模型在所有数据集上的平均指标")
    print(f"{'─' * 130}")
    models = df['Model'].unique()
    sum_header = f"{'Model':<28s}"
    for c in metric_cols:
        if c in df.columns:
            sum_header += f" {c:>8s}"
    print(sum_header)
    print("-" * 130)

    for m_name in models:
        sub = df[df['Model'] == m_name]
        line = f"{m_name:<28s}"
        for c in metric_cols:
            if c in df.columns:
                vals = sub[c]
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    avg = np.mean(numeric_vals)
                    line += f" {avg:>8.2f}" if c not in ("Time(s)", "PSNR") else f" {avg:>8.3f}"
                else:
                    line += f" {'N/A':>8s}"
        print(line)

    print(f"{'=' * 130}\n")


def _parse_args():
    """CLI 参数解析。"""
    parser = argparse.ArgumentParser(
        description="LAMF-Fuse 全自动评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python eval_all.py                                          # 使用脚本内 CONFIG
  python eval_all.py --ckpt models/new.pth --name MyNewModel  # 评测单个新模型
  python eval_all.py --test_root ./my_test_data               # 自定义测试集路径
  python eval_all.py --keep                                   # 保留融合图像不删除
  python eval_all.py --no-flops                               # 跳过 FLOPs 计算
        """
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="单个模型权重路径 (覆盖 CKPT_LIST)")
    parser.add_argument("--name", type=str, default=None,
                        help="模型名称 (配合 --ckpt 使用)")
    parser.add_argument("--test_root", type=str, default=None,
                        help="测试集根目录 (覆盖 TEST_ROOT)")
    parser.add_argument("--keep", action="store_true",
                        help="保留融合图像")
    parser.add_argument("--no-flops", action="store_true",
                        help="跳过 FLOPs/参数量计算")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["A", "C", "D", "D+", "E"],
                        help="消融变体: A/C/D/D+/E (默认 E, 自动匹配 Base/Detail 架构)")
    return parser.parse_args()


# ============================================================
if __name__ == "__main__":
    evaluate_all()
