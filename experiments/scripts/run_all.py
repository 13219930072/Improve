#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
LAMF-Fuse 一键训练 + 评估流水线
===============================================================================
功能:
  1. 一键运行 训练(train_my.py) → 评估(eval_all.py)
  2. 自动定位训练产出的模型权重
  3. 支持仅训练 / 仅评估 / 全流程 三种模式
  4. 训练日志 + 评估结果自动归档
  5. 多数据集自动评测 + Excel 输出 + 融合图自动清理

用法:
  python run_all.py                    # 训练 + 评估 (全流程)
  python run_all.py --train-only       # 仅训练
  python run_all.py --eval-only        # 仅评估 (需指定 --ckpt)
  python run_all.py --ckpt models/xxx.pth  # 跳过训练，直接用该权重评估
  python run_all.py --no-eval          # 训练后不评估
  python run_all.py --keep-fused       # 评估后保留融合图像

===============================================================================
"""

import os
import sys
import time
import glob
import argparse
import subprocess
from datetime import datetime

#CUDA内存碎片化防护
CUDA_ALLOC_CONF = 'max_split_size_mb:128'

# ============================================================
# ==================  统 一 配 置 区 域  ======================
# ============================================================

# ── 项目路径 ──
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJ_ROOT, "train_my.py")
EVAL_SCRIPT = os.path.join(PROJ_ROOT, "eval_all.py")

# ── 模型存档目录 ──
MODEL_SEARCH_DIRS = [
    os.path.join(PROJ_ROOT, "models"),
    "/home/hyhz/HT/cddfuse_data/models",
]

# ── 测试集根目录 (传给 eval_all.py) ──
TEST_ROOT = "/home/hyhz/HT/test_img"

# ── 日志与结果存档 ──
LOG_DIR = os.path.join(PROJ_ROOT, "run_logs")        # 训练日志存档
RESULT_DIR = os.path.join(PROJ_ROOT, "eval_results") # 评估 Excel 存档 (eval_all.py 默认)

# ============================================================
# ==================  工 具 函 数  ============================
# ============================================================

def timestamp_str():
    return datetime.now().strftime("%m%d-%H%M%S")


def find_latest_checkpoint(dirs, prefer_final=True):
    """
    在给定目录列表中查找最新的 .pth 权重文件。
    prefer_final: 优先选择文件名含 "Final" 的权重。
    返回: (ckpt_path, model_name) 或 (None, None)
    """
    candidates = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.pth")):
            candidates.append(f)
        # 也搜一下子目录
        for f in glob.glob(os.path.join(d, "*", "*.pth")):
            candidates.append(f)

    if not candidates:
        return None, None

    # 如果有 "Final" 文件，优先选最新的
    finals = [f for f in candidates if "Final" in os.path.basename(f)]
    if finals and prefer_final:
        latest = max(finals, key=os.path.getmtime)
    else:
        latest = max(candidates, key=os.path.getmtime)

    # 提取模型名: 取文件名 (不含扩展名)
    model_name = os.path.splitext(os.path.basename(latest))[0]
    return latest, model_name


def list_checkpoints(dirs):
    """列出所有可用权重。"""
    ckpts = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.pth")):
            ckpts.append((os.path.basename(f), f, os.path.getmtime(f)))
    ckpts.sort(key=lambda x: x[2], reverse=True)
    return ckpts


# ============================================================
# ==================  阶 段 1 : 训 练  ========================
# ============================================================

def run_training():
    """
    启动 train_my.py 子进程进行训练。
    返回: (success: bool, log_path: str)
    """
    print("\n" + "=" * 80)
    print("  阶段 1/2: 训练 (train_my.py)")
    print("=" * 80)

    ts = timestamp_str()
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"train_{ts}.log")

    print(f"  训练脚本: {TRAIN_SCRIPT}")
    print(f"  日志文件: {log_path}")
    print(f"  Python:   {sys.executable}")
    print("-" * 80)

    # 记录训练开始前的已有模型 (用于后续发现新模型)
    before_ckpts = set()
    for d in MODEL_SEARCH_DIRS:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.pth")):
                before_ckpts.add(f)
            for f in glob.glob(os.path.join(d, "*", "*.pth")):
                before_ckpts.add(f)

    # 启动训练子进程
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'  # 实时输出
    env['PYTORCH_CUDA_ALLOC_CONF'] = CUDA_ALLOC_CONF
    proc = None

    try:
        with open(log_path, 'w', encoding='utf-8') as log_f:
            proc = subprocess.Popen(
                [sys.executable, "-u", TRAIN_SCRIPT],
                cwd=PROJ_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )

            # 实时打印 + 写入日志
            for line in proc.stdout:
                print(f"  {line}", end='')
                log_f.write(line)
                log_f.flush()

            proc.wait()

        if proc.returncode == 0:
            print("\n" + "=" * 80)
            print("  ✓ 训练完成！")
            print("=" * 80)

            # 发现新产出的模型
            after_ckpts = set()
            for d in MODEL_SEARCH_DIRS:
                if os.path.isdir(d):
                    for f in glob.glob(os.path.join(d, "*.pth")):
                        after_ckpts.add(f)
                    for f in glob.glob(os.path.join(d, "*", "*.pth")):
                        after_ckpts.add(f)
            new_ckpts = after_ckpts - before_ckpts
            if new_ckpts:
                print(f"  新生成权重: {len(new_ckpts)} 个")
                for c in sorted(new_ckpts):
                    print(f"    → {c}")

            return True, log_path
        else:
            print(f"\n  ✗ 训练失败！返回码: {proc.returncode}")
            print(f"  完整日志: {log_path}")
            return False, log_path

    except KeyboardInterrupt:
        print("\n  ⚠ 用户中断训练")
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        return False, log_path
    except Exception as e:
        print(f"\n  ✗ 训练异常: {e}")
        if proc is not None:
            proc.terminate()
        return False, log_path


# ============================================================
# ==================  阶 段 2 : 评 估  ========================
# ============================================================

def run_evaluation(ckpt_path, model_name=None, keep_fused=False):
    """
    调用 eval_all.py 进行全数据集评估。
    返回: success: bool
    """
    print("\n" + "=" * 80)
    print("  阶段 2/2: 评估 (eval_all.py)")
    print("=" * 80)

    if not os.path.exists(ckpt_path):
        print(f"  ✗ 权重文件不存在: {ckpt_path}")
        return False

    if model_name is None:
        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    print(f"  模型名称: {model_name}")
    print(f"  权重路径: {ckpt_path}")
    print(f"  测试根目录: {TEST_ROOT}")
    print("-" * 80)

    # 构建命令行参数
    cmd = [
        sys.executable, "-u", EVAL_SCRIPT,
        "--ckpt", ckpt_path,
        "--name", model_name,
        "--test_root", TEST_ROOT,
    ]
    if keep_fused:
        cmd.append("--keep")

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJ_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(f"  {line}", end='')

        proc.wait()

        if proc.returncode == 0:
            print("\n" + "=" * 80)
            print("  ✓ 评估完成！")
            print(f"  Excel 结果目录: {os.path.abspath(RESULT_DIR)}")
            print("=" * 80)
            return True
        else:
            print(f"\n  ✗ 评估失败！返回码: {proc.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n  ⚠ 用户中断评估")
        if proc is not None:
            proc.terminate()
        return False
    except Exception as e:
        print(f"\n  ✗ 评估异常: {e}")
        if proc is not None:
            proc.terminate()
        return False


# ============================================================
# ==================  主 控 流 程  ============================
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LAMF-Fuse 一键训练+评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_all.py                           # 训练 + 评估 (全流程)
  python run_all.py --train-only              # 仅训练
  python run_all.py --eval-only               # 仅评估 (自动找最新权重)
  python run_all.py --ckpt models/my.pth      # 用指定权重评估
  python run_all.py --no-eval                 # 训练后跳过评估
  python run_all.py --keep-fused              # 保留融合图像不删除
        """
    )
    parser.add_argument("--train-only", action="store_true",
                        help="仅执行训练")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅执行评估")
    parser.add_argument("--no-eval", action="store_true",
                        help="训练后跳过评估")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="指定权重文件进行评估 (跳过自动发现)")
    parser.add_argument("--name", type=str, default=None,
                        help="模型名称 (配合 --ckpt 使用)")
    parser.add_argument("--keep-fused", action="store_true",
                        help="评估后保留融合图像")
    parser.add_argument("--list-models", action="store_true",
                        help="列出所有可用权重后退出")

    args = parser.parse_args()

    # ── 列出可用模型 ──
    if args.list_models:
        ckpts = list_checkpoints(MODEL_SEARCH_DIRS)
        if not ckpts:
            print("未找到任何 .pth 权重文件")
        else:
            print(f"\n找到 {len(ckpts)} 个权重文件:\n")
            for i, (name, path, mtime) in enumerate(ckpts):
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  [{i+1}] {name}")
                print(f"      路径: {path}")
                print(f"      时间: {mtime_str}\n")
        return

    # ── 确定运行模式 ──
    do_train = True
    do_eval = True

    if args.train_only:
        do_eval = False
    if args.eval_only:
        do_train = False
    if args.no_eval:
        do_eval = False

    # ── 打印配置 ──
    print("\n" + "=" * 80)
    print("  LAMF-Fuse 一键训练+评估流水线")
    print("=" * 80)
    print(f"  项目根目录: {PROJ_ROOT}")
    print(f"  Python:     {sys.executable}")
    print(f"  运行模式:   {'训练' if do_train else ''}{' + ' if do_train and do_eval else ''}{'评估' if do_eval else ''}")
    print(f"  搜索目录:   {MODEL_SEARCH_DIRS}")
    print(f"  测试数据:   {TEST_ROOT}")
    print(f"  开始时间:   {timestamp_str()}")
    print("=" * 80)

    t_start = time.time()

    # ════════════════════════════════════════════════════════
    # 阶段 1: 训练
    # ════════════════════════════════════════════════════════
    ckpt_path = args.ckpt
    model_name = args.name

    if do_train:
        success, train_log = run_training()
        if not success:
            print("\n  ✗ 训练失败，终止流水线。")
            print(f"  训练日志: {train_log}")
            sys.exit(1)
        # 训练成功后，不显式指定 ckpt，后面自动发现
        ckpt_path = None
        model_name = None

    # ════════════════════════════════════════════════════════
    # 阶段 2: 评估
    # ════════════════════════════════════════════════════════
    if do_eval:
        # 自动发现权重
        if ckpt_path is None:
            ckpt_path, model_name = find_latest_checkpoint(MODEL_SEARCH_DIRS, prefer_final=True)
            if ckpt_path is None:
                print("\n  ✗ 未找到任何 .pth 权重文件，无法进行评估。")
                print(f"  请在以下目录放置权重: {MODEL_SEARCH_DIRS}")
                sys.exit(1)
            print(f"\n  [自动发现] 使用最新权重: {model_name}")
            print(f"  [自动发现] 路径: {ckpt_path}")

        success = run_evaluation(ckpt_path, model_name, keep_fused=args.keep_fused)
        if not success:
            print("\n  ✗ 评估失败！")
            sys.exit(1)

    # ── 完成 ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ✓ 流水线全部完成！")
    print(f"  总耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)")
    if do_train:
        print(f"  训练日志: {LOG_DIR}")
    if do_eval:
        print(f"  评估结果: {os.path.abspath(RESULT_DIR)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
