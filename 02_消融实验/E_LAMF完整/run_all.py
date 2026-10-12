# -*- coding: utf-8 -*-
"""
===========================================================================
LAMF Full — 一键训练 + 评估
===========================================================================
用法:
  python run_all.py                 # 训练 + 评估 (全流程)
  python run_all.py --train-only    # 仅训练
  python run_all.py --eval-only     # 仅评估 (自动找最新权重)
  python run_all.py --ckpt checkpoints/xxx.pth  # 用指定权重评估
  python run_all.py --keep-fused    # 保留融合图像

输出:
  - 训练权重 → checkpoints/
  - 评估结果 → results/eval_xxx.xlsx
  - 实验报告 → results/实验报告_xxx.xlsx (含详细备注)
===========================================================================
"""
import os
import sys
import time
import glob
import argparse
from datetime import datetime

# ---- 路径 ----
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), '01_模型代码'))
from common import filtered_popen
PROJ_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
TRAIN_SCRIPT = os.path.join(EXP_DIR, "train.py")
EVAL_SCRIPT = os.path.join(PROJ_ROOT, "eval_all.py")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
RESULT_DIR = os.path.join(EXP_DIR, "results")

# ---- 实验配置 (写入报告) ----
EXP_CONFIG = {
    "实验编号": "E (LAMF 最终模型)",
    "消融变体": "E",
    "Base融合层": "LocalEnhancedVSSBlock (LESS-Mamba with SimpleGate)",
    "Detail融合层": "NAFDetailBlock (SimpleGate gating)",
    "损失函数": "Fusionloss only (intensity + gradient + decomp), 无Mask, 无MI Loss",
    "训练数据": "MSRS_train_imgsize_128_stride_200.h5",
    "训练epochs": "120 (epoch_gap=40)",
    "Phase I": "Epoch 1-40: 自编码器预训练",
    "Phase II": "Epoch 41-120: 融合网络训练",
    "Batch size": "8",
    "学习率": "1e-4, StepLR(step=20, gamma=0.5)",
    "梯度裁剪": "max_norm=0.01",
    "损失系数": "mse_VF=1, mse_IF=1, decomp=2, tv=5",
    "测试数据集": "TNO / RoadScene / M3FD / FMB",
    "FLOPs输入尺寸": "128 × 128",
}

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
def timestamp():
    return datetime.now().strftime("%m%d-%H%M")


def find_latest_checkpoint():
    """在 checkpoints/ 下找最新的 .pth"""
    candidates = []
    for f in glob.glob(os.path.join(CKPT_DIR, "*.pth")):
        candidates.append(f)
    if not candidates:
        return None, None
    latest = max(candidates, key=os.path.getmtime)
    name = os.path.splitext(os.path.basename(latest))[0]
    return latest, name


# ============================================================
# 阶段 1: 训练
# ============================================================
def run_training():
    print("\n" + "=" * 70)
    print("  阶段 1/2: 训练 (LAMF Full, 无 Mask)")
    print("=" * 70)
    print(f"  训练脚本: {TRAIN_SCRIPT}")
    print(f"  权重目录: {CKPT_DIR}")

    # 记录训练前已有权重
    before = set(glob.glob(os.path.join(CKPT_DIR, "*.pth")))

    ret = filtered_popen([sys.executable, "-u", TRAIN_SCRIPT], cwd=EXP_DIR)

    if ret != 0:
        print(f"\n  ✗ 训练失败！返回码: {ret}")
        return False

    # 发现新权重
    after = set(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
    new_ckpts = after - before
    if new_ckpts:
        print(f"\n  ✓ 训练完成! 新生成权重:")
        for c in sorted(new_ckpts):
            print(f"    → {c}")
    return True


# ============================================================
# 阶段 2: 评估
# ============================================================
def run_evaluation(ckpt_path, model_name, keep_fused=False):
    print("\n" + "=" * 70)
    print("  阶段 2/2: 评估")
    print("=" * 70)
    print(f"  权重: {ckpt_path}")
    print(f"  模型名: {model_name}")

    cmd = [
        sys.executable, "-u", EVAL_SCRIPT,
        "--ckpt", ckpt_path,
        "--name", model_name,
        "--variant", "E",
    ]
    if keep_fused:
        cmd.append("--keep")

    ret = filtered_popen(cmd, cwd=PROJ_ROOT)
    return ret == 0


# ============================================================
# 阶段 3: 生成实验报告
# ============================================================
def generate_report(model_name, ckpt_path, eval_excel_path):
    """
    读取 eval_all.py 产出的 Excel，附加实验配置信息，
    生成最终报告到 results/实验报告_xxx.xlsx
    """
    try:
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  ⚠ openpyxl/pandas 未安装，跳过报告生成")
        return None

    # 尝试读取 eval 产出的 Excel
    df_metrics = None
    if eval_excel_path and os.path.exists(eval_excel_path):
        try:
            df_metrics = pd.read_excel(eval_excel_path, sheet_name=0)
        except Exception:
            pass

    wb = Workbook()

    # ── 辅助样式 ──
    title_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    title_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    header_font = Font(name="微软雅黑", size=10, bold=True)
    data_font = Font(name="微软雅黑", size=10)
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    avg_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    # ================================================================
    # Sheet 1: 评估指标
    # ================================================================
    ws1 = wb.active
    ws1.title = "评估指标"

    if df_metrics is not None and not df_metrics.empty:
        # 写表头
        for col_idx, col_name in enumerate(df_metrics.columns, 1):
            cell = ws1.cell(row=1, column=col_idx, value=col_name)
            cell.font = title_font
            cell.fill = title_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = thin_border

        # 写数据行
        for row_idx, (_, row) in enumerate(df_metrics.iterrows()):
            for col_idx, val in enumerate(row, 1):
                cell = ws1.cell(row=row_idx + 2, column=col_idx, value=val)
                cell.font = data_font
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border

        # 列宽
        for col_idx in range(1, len(df_metrics.columns) + 1):
            ws1.column_dimensions[get_column_letter(col_idx)].width = 14
    else:
        ws1.cell(row=1, column=1, value="评估数据未找到，请检查 eval_all.py 输出").font = data_font

    # ================================================================
    # Sheet 2: 实验配置
    # ================================================================
    ws2 = wb.create_sheet("实验配置")

    ws2.merge_cells('A1:B1')
    cell = ws2.cell(row=1, column=1, value=f"LAMF 实验 E — 训练与评估配置")
    cell.font = Font(name="微软雅黑", size=14, bold=True)
    cell.alignment = Alignment(horizontal='center')

    ws2.merge_cells('A2:B2')
    ws2.cell(row=2, column=1,
             value=f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                   f"权重: {os.path.basename(ckpt_path)}").font = Font(name="微软雅黑", size=9, color="888888")

    row = 4
    for key, val in EXP_CONFIG.items():
        cell_k = ws2.cell(row=row, column=1, value=key)
        cell_k.font = Font(name="微软雅黑", size=10, bold=True)
        cell_k.fill = header_fill
        cell_k.border = thin_border

        cell_v = ws2.cell(row=row, column=2, value=val)
        cell_v.font = data_font
        cell_v.border = thin_border
        row += 1

    # 额外的动态信息
    extra_info = {
        "权重文件": os.path.basename(ckpt_path),
        "权重路径": ckpt_path,
        "模型名称": model_name,
    }
    row += 1
    for key, val in extra_info.items():
        cell_k = ws2.cell(row=row, column=1, value=key)
        cell_k.font = Font(name="微软雅黑", size=10, bold=True)
        cell_k.fill = header_fill
        cell_k.border = thin_border
        cell_v = ws2.cell(row=row, column=2, value=val)
        cell_v.font = data_font
        cell_v.border = thin_border
        row += 1

    ws2.column_dimensions['A'].width = 26
    ws2.column_dimensions['B'].width = 70

    # ================================================================
    # Sheet 3: 消融框架
    # ================================================================
    ws3 = wb.create_sheet("消融框架")

    ws3.merge_cells('A1:G1')
    ws3.cell(row=1, column=1,
             value="消融实验架构对比 (★ = 本次实验)").font = Font(name="微软雅黑", size=12, bold=True)

    arch_headers = ["实验", "Base融合层", "Detail融合层", "Base参数", "Detail参数", "总参数(M)", "说明"]
    for col_idx, h in enumerate(arch_headers, 1):
        cell = ws3.cell(row=3, column=col_idx, value=h)
        cell.font = title_font
        cell.fill = title_fill
        cell.alignment = Alignment(horizontal='center')
        cell.border = thin_border

    arch_data = [
        ["A",  "BaseFeatureExtraction\n(Transformer + Attention)", "SimpleDetailFusion\n(3层卷积, LeakyReLU)", "363K", "45K", "~0.87", "CDDFuse 基线"],
        ["C",  "BaseFeatureExtraction\n(Transformer + Attention)", "DetailFeatureExtraction\n(NAFDetailBlock, SG)", "363K", "18K", "~0.85", "+SimpleGate (Detail分支)"],
        ["D",  "OriginalVSSBlock\n(原生MambaIR, SiLU)", "SimpleDetailFusion\n(3层卷积, LeakyReLU)", "46K", "45K", "~0.83", "+原生MambaIR (Base分支)"],
        ["D+", "OriginalVSSBlock\n(原生MambaIR, SiLU)", "DetailFeatureExtraction\n(NAFDetailBlock, SG)", "46K", "18K", "~0.83", "桥接消融: D→E"],
        ["★E","LocalEnhancedVSSBlock\n(LESS-Mamba, SG)", "DetailFeatureExtraction\n(NAFDetailBlock, SG)", "46K", "18K", "~0.83", "LAMF 最终模型\n(本次实验)"],
    ]
    for row_idx, row_data in enumerate(arch_data, 4):
        is_current = row_data[0].startswith("★")
        for col_idx, val in enumerate(row_data, 1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=val)
            cell.font = Font(name="微软雅黑", size=10, bold=is_current)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = thin_border
            if is_current:
                cell.fill = avg_fill

    # 行高与列宽
    for r in range(4, 4 + len(arch_data)):
        ws3.row_dimensions[r].height = 50
    for col_idx, w in enumerate([8, 28, 28, 12, 12, 12, 22], 1):
        ws3.column_dimensions[get_column_letter(col_idx)].width = w

    # ================================================================
    # 保存
    # ================================================================
    report_path = os.path.join(RESULT_DIR, f"实验报告_{model_name}_{timestamp()}.xlsx")
    wb.save(report_path)
    return report_path


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="LAMF Full 一键训练+评估")
    parser.add_argument("--train-only", action="store_true", help="仅训练")
    parser.add_argument("--eval-only", action="store_true", help="仅评估 (自动找最新权重)")
    parser.add_argument("--ckpt", type=str, default=None, help="指定权重进行评估")
    parser.add_argument("--name", type=str, default=None, help="模型名称 (配合 --ckpt)")
    parser.add_argument("--keep-fused", action="store_true", help="保留融合图像")
    args = parser.parse_args()

    do_train = not args.eval_only
    do_eval = not args.train_only

    print("\n" + "=" * 70)
    print("  LAMF Full — 一键训练 + 评估")
    print("=" * 70)
    print(f"  实验目录: {EXP_DIR}")
    print(f"  模式: {'训练' if do_train else ''}{' + ' if do_train and do_eval else ''}{'评估' if do_eval else ''}")
    print(f"  权重目录: {CKPT_DIR}")
    print(f"  结果目录: {RESULT_DIR}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t_start = time.time()
    ckpt_path = args.ckpt
    model_name = args.name

    # ── 阶段 1: 训练 ──
    if do_train:
        success = run_training()
        if not success:
            print("\n  ✗ 训练失败，终止。")
            sys.exit(1)
        ckpt_path = None  # 训练成功后自动发现

    # ── 阶段 2: 评估 ──
    eval_excel = None
    if do_eval:
        if ckpt_path is None:
            ckpt_path, model_name = find_latest_checkpoint()
            if ckpt_path is None:
                print(f"\n  ✗ 在 {CKPT_DIR} 下未找到任何 .pth 文件")
                sys.exit(1)
            print(f"\n  [自动发现] 使用最新权重: {model_name}")
            print(f"  [自动发现] 路径: {ckpt_path}")

        if model_name is None:
            model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

        success = run_evaluation(ckpt_path, model_name, keep_fused=args.keep_fused)
        if not success:
            print("\n  ✗ 评估失败！")
            sys.exit(1)

        # 找到 eval_all.py 产出的 Excel
        eval_dir = os.path.join(PROJ_ROOT, "eval_results")
        excel_files = sorted(glob.glob(os.path.join(eval_dir, "eval_*.xlsx")), key=os.path.getmtime, reverse=True)
        if excel_files:
            eval_excel = excel_files[0]
            print(f"\n  [评估Excel] {eval_excel}")

    # ── 阶段 3: 生成实验报告 ──
    if do_eval and ckpt_path:
        print(f"\n  生成实验报告...")
        report_path = generate_report(model_name, ckpt_path, eval_excel)
        if report_path:
            print(f"  ✓ 实验报告: {report_path}")

    # ── 完成 ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  ✓ 全流程完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"  权重: {CKPT_DIR}")
    print(f"  结果: {RESULT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
