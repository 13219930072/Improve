# -*- coding: utf-8 -*-
"""
===========================================================================
消融实验一键批量运行
===========================================================================
用法:
  python run_all_experiments.py                  # 按推荐顺序跑全部 (A eval → C → D → D+)
  python run_all_experiments.py --eval-only      # 仅评估 (需要已有权重)
  python run_all_experiments.py --train-only     # 仅训练
  python run_all_experiments.py --skip A         # 跳过 A
  python run_all_experiments.py --only C,D_plus  # 仅跑指定实验
  python run_all_experiments.py --keep-fused     # 保留融合图像
  python run_all_experiments.py --stop-on-error  # 遇错即停 (默认继续)

输出:
  每个实验的结果在各自文件夹的 results/ 下:
    A/results/实验报告_xxx.xlsx
    C/results/实验报告_xxx.xlsx`
    D/results/实验报告_xxx.xlsx
    D_plus/results/实验报告_xxx.xlsx
===========================================================================
"""
import os
import sys
import time
import argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from common import filtered_popen

# ---- 路径 ----
EXP_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 实验定义 (按运行顺序) ----
# 每个实验: (文件夹名, 显示名, 需要训练, variant)
EXPERIMENTS = [
    ("A",      "A  (CDDFuse 基线)",          False, "A"),
    ("C",      "C  (+SimpleGate)",             True,  "C"),
    ("D",      "D  (+原生MambaIR)",            True,  "D"),
    ("D_plus", "D+ (桥接消融, LESS-Mamba核心)", True,  "D+"),
]


def timestamp():
    return datetime.now().strftime("%m%d-%H%M")


def run_one(exp_folder, exp_label, needs_train, variant, do_train, do_eval,
            keep_fused, stop_on_error):
    """
    运行单个实验的 run_all.py。
    返回: (success: bool, elapsed_min: float)
    """
    exp_dir = os.path.join(EXP_ROOT, exp_folder)
    run_all = os.path.join(exp_dir, "run_all.py")

    if not os.path.exists(run_all):
        print(f"  ✗ {run_all} 不存在，跳过")
        return False, 0

    cmd = [sys.executable, "-u", run_all]

    if not needs_train:
        # 已有权重的实验 (A): 仅在显式要求训练时才传 --train
        if do_train and not do_eval:
            cmd.append("--train")
        # 否则走 run_all.py 默认逻辑: eval-only
    else:
        # C/D/D+: 需要训练
        if do_train and not do_eval:
            cmd.append("--train-only")
        elif do_eval and not do_train:
            cmd.append("--eval-only")

    if keep_fused:
        cmd.append("--keep-fused")

    print("\n" + "=" * 80)
    print(f"  运行: {exp_label}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"  开始: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)

    t0 = time.time()

    ret = filtered_popen(cmd, cwd=exp_dir)
    elapsed = (time.time() - t0) / 60.0
    success = ret == 0

    status = "✓ 完成" if success else f"✗ 失败 (返回码 {ret})"
    print(f"\n  {status}  |  耗时: {elapsed:.1f} 分钟")

    return success, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="消融实验批量运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all_experiments.py                     # 全部: A eval + C/D/D+ 训练+评估
  python run_all_experiments.py --eval-only         # 仅评估已有权重
  python run_all_experiments.py --only C,D_plus     # 仅跑 C 和 D+
  python run_all_experiments.py --skip A,D          # 跳过 A 和 D
        """
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估 (需要权重已存在)")
    parser.add_argument("--train-only", action="store_true",
                        help="仅训练 (不评估)")
    parser.add_argument("--only", type=str, default=None,
                        help="仅运行指定实验 (逗号分隔, 如: C,D_plus)")
    parser.add_argument("--skip", type=str, default=None,
                        help="跳过指定实验 (逗号分隔, 如: A,D)")
    parser.add_argument("--keep-fused", action="store_true",
                        help="保留融合图像")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="遇错即停 (默认继续运行下一个)")
    args = parser.parse_args()

    # 解析筛选条件
    only_set = set(args.only.split(",")) if args.only else None
    skip_set = set(args.skip.split(",")) if args.skip else set()

    do_train = not args.eval_only
    do_eval = not args.train_only

    # 筛选要运行的实验
    to_run = []
    for folder, label, needs_train, variant in EXPERIMENTS:
        if only_set and folder not in only_set:
            continue
        if folder in skip_set:
            continue
        # 如果仅评估但实验需要训练且没有权重，仍然尝试（让 run_all.py 报错）
        to_run.append((folder, label, needs_train, variant))

    if not to_run:
        print("没有需要运行的实验。")
        return

    print("=" * 80)
    print("  消融实验批量运行")
    print("=" * 80)
    print(f"  模式: {'训练' if do_train else ''}{' + ' if do_train and do_eval else ''}{'评估' if do_eval else ''}")
    print(f"  计划运行: {len(to_run)} 个实验")
    for folder, label, needs_train, variant in to_run:
        if not needs_train and do_train and not do_eval:
            action = "train"
        elif not needs_train:
            action = "eval only"  # 已有权重，默认只评估
        elif do_train and do_eval:
            action = "train+eval"
        elif do_train:
            action = "train"
        else:
            action = "eval"
        print(f"    {label:<40s} [{action}]")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t_total = time.time()
    results = []

    for folder, label, needs_train, variant in to_run:
        ok, elapsed = run_one(folder, label, needs_train, variant,
                              do_train, do_eval,
                              args.keep_fused, args.stop_on_error)
        results.append((label, ok, elapsed))
        if not ok and args.stop_on_error:
            print(f"\n  ⚠ 遇错即停，剩余实验跳过。")
            break

    # ── 汇总 ──
    total_min = (time.time() - t_total) / 60.0
    print("\n" + "=" * 80)
    print("  批量运行汇总")
    print("=" * 80)
    success_count = 0
    for label, ok, elapsed in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {label:<40s} {elapsed:>8.1f} min")
        if ok:
            success_count += 1
    print(f"  {'─' * 60}")
    print(f"  完成: {success_count}/{len(results)}  |  总耗时: {total_min:.1f} 分钟")
    print(f"  各实验结果分别在:")
    for folder, _, _, _ in to_run:
        rdir = os.path.join(EXP_ROOT, folder, "results")
        if os.path.isdir(rdir):
            print(f"    → {folder}/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
