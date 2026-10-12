# -*- coding: utf-8 -*-
"""
===========================================================================
实验公共模块
===========================================================================
提供消融实验中各变体所需的共享组件，避免重复代码。
"""

import sys
import subprocess
import torch
import torch.nn as nn


class SimpleDetailFusion(nn.Module):
    """
    极简 Detail 融合层 —— 模拟 CDDFuse 原版的 Detail 融合方式。

    CDDFuse 原版中 Detail 分支使用轻量级卷积融合（不含 INN 可逆耦合层
    也不含 SimpleGate 门控），作为 Detail 融合的 baseline。

    对比:
      SimpleDetailFusion  → 实验 A / D (无门控 baseline)
      NAFDetailBlock(SG)  → 实验 C / D+ / E (SimpleGate 增强)
    """
    def __init__(self, dim=64):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.fusion(x) + x  # 残差连接


# ============================================================================
# 子进程输出过滤 —— 让训练/评估只显示关键摘要行
# ============================================================================

def filtered_popen(cmd, cwd, prefix="  "):
    """
    运行子进程并智能过滤输出:
      - 带 \\r 的进度行 → 折叠为一行原地刷新 (如 per-batch loss, tqdm)
      - 普通行 → 正常打印
      - 评估指标行 → 保留并高亮
      - 错误/警告 → 始终显示
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    current_progress = ""
    remaining = ""

    while True:
        chunk = proc.stdout.read(256)
        if not chunk:
            break
        remaining += chunk

        while '\n' in remaining:
            line, remaining = remaining.split('\n', 1)

            if '\r' in line:
                # 进度行 (per-batch loss / tqdm) → 只保留最后一段，原地刷新
                last = line.split('\r')[-1].strip()
                if last and last != current_progress:
                    current_progress = last
                    sys.stdout.write(f"\r{prefix}{last}")
                    sys.stdout.flush()
            elif line.strip():
                # 普通行 → 先结束进度行 (如果有)，再打印
                if current_progress:
                    sys.stdout.write("\n")
                    current_progress = ""
                sys.stdout.write(f"{prefix}{line}\n")
                sys.stdout.flush()

    # 处理尾部残留
    if remaining.strip():
        if '\r' in remaining:
            last = remaining.split('\r')[-1].strip()
            if last:
                sys.stdout.write(f"\r{prefix}{last}")
        else:
            sys.stdout.write(f"{prefix}{remaining}")
        sys.stdout.flush()

    if current_progress:
        sys.stdout.write("\n")
    sys.stdout.flush()

    proc.wait()
    return proc.returncode


# ============================================================================
# 消融实验配置速查表
# ============================================================================
#
#  实验   Base融合层                  Detail融合层              门控
#  ───── ────────────────────────── ────────────────────────── ──────
#   A     BaseFeatureExtraction      SimpleDetailFusion         SiLU
#   C     BaseFeatureExtraction      DetailFeatureExtraction    SG
#   D     OriginalVSSBlock           SimpleDetailFusion         SiLU
#   D+    OriginalVSSBlock           DetailFeatureExtraction    SiLU/SG
#   E     LocalEnhancedVSSBlock      DetailFeatureExtraction    SG
#
#  配置函数（导入即用）：
#    make_base_fusion(exp_id)       → nn.Module
#    make_detail_fusion(exp_id)     → nn.Module
# ============================================================================
