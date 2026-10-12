# -*- coding: utf-8 -*-
"""
===========================================================================
lamf_cpu.py — LAMF 完整网络（CPU/ARM 推理版）
===========================================================================
复用 net.py 的 Restormer Encoder/Decoder 与 NAFDetailBlock（纯 PyTorch，
无 CUDA 依赖），仅把 Base 融合层（LESS-Mamba）替换为 mamba_cpu 版本。

权重兼容: 现有 .pth 的 state_dict key 与本实现完全一致，
        只需去除训练时 DataParallel 产生的 'module.' 前缀即可加载。

用法:
    from lamf_cpu import LAMF_CPU
    model = LAMF_CPU()
    model.load_lamf_weights('/path/to/LAMF_Full_Final_07-17-13-19.pth')
    fused = model(ir_tensor, vis_tensor)   # 各 (1,1,H,W), 值域 [0,1]
===========================================================================
"""
import os
import sys

import torch
import torch.nn as nn

# 让本文件可独立运行：把项目根目录加入 path 以导入 net.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction  # noqa: E402
from mamba_cpu import LocalEnhancedVSSBlockCPU  # noqa: E402


def strip_dataparallel_prefix(state_dict):
    """去除 DataParallel 训练产生的 'module.' 前缀"""
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v
            for k, v in state_dict.items()}


class LAMF_CPU(nn.Module):
    """
    LAMF 完整推理网络（CPU 版）

    结构（与训练时一致）:
        Encoder(共享) → Base特征 / Detail特征
        Base特征 → LESS-Mamba 融合 (LocalEnhancedVSSBlockCPU)
        Detail特征 → NAFDetailBlock 融合 (DetailFeatureExtraction)
        → Decoder → 融合图像
    """
    def __init__(self, dim=64, use_chunked=False, chunk_size=512):
        super().__init__()
        self.encoder = Restormer_Encoder()
        self.decoder = Restormer_Decoder()
        self.base_fuse = LocalEnhancedVSSBlockCPU(dim=dim, use_chunked=use_chunked, chunk_size=chunk_size)
        self.detail_fuse = DetailFeatureExtraction(num_layers=1, dim=dim)

    def forward(self, ir, vis):
        """ir/vis: (B,1,H,W) float32, 值域 [0,1] → 融合图 (B,1,H,W)"""
        fV_B, fV_D, fV = self.encoder(vis)
        fI_B, fI_D, fI = self.encoder(ir)
        fF_B = self.base_fuse(fI_B + fV_B)
        fF_D = self.detail_fuse(fI_D + fV_D)
        fused, _ = self.decoder(vis, fF_B, fF_D)
        return fused

    def load_lamf_weights(self, ckpt_path, verbose=True):
        """加载 LAMF 权重（自动处理 module. 前缀），返回加载报告"""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        report = {}
        mapping = [
            ('encoder', 'DIDF_Encoder'),
            ('decoder', 'DIDF_Decoder'),
            ('base_fuse', 'BaseFuseLayer'),
            ('detail_fuse', 'DetailFuseLayer'),
        ]
        for my_name, ckpt_key in mapping:
            module = getattr(self, my_name)
            sd = strip_dataparallel_prefix(ckpt[ckpt_key])
            missing, unexpected = module.load_state_dict(sd, strict=False)
            report[my_name] = {
                'missing': list(missing),
                'unexpected': list(unexpected),
                'num_tensors': len(sd),
            }
            if verbose:
                status = '✅' if (not missing and not unexpected) else '⚠️'
                print(f"  {status} {my_name:12s} ← {ckpt_key:16s} "
                      f"({len(sd)} 个张量, 缺失 {len(missing)}, 多余 {len(unexpected)})")
        all_ok = all(not r['missing'] and not r['unexpected'] for r in report.values())
        self._load_report = report
        if verbose:
            print("✅ 权重加载完毕（严格匹配）" if all_ok else "⚠️ 权重加载存在不匹配，请检查")
        return report


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total


if __name__ == '__main__':
    print("=" * 70)
    print("LAMF_CPU 构建与权重加载自测")
    print("=" * 70)

    ckpt_path = os.path.join(_PROJ_ROOT, 'experiments', 'LAMF_Full', 'checkpoints',
                             'LAMF_Full_Final_07-17-13-19.pth')
    model = LAMF_CPU()
    model.eval()
    print(f"模型参数量: {count_params(model):,} ({count_params(model)/1e6:.4f} M)")
    print("加载权重:")
    report = model.load_lamf_weights(ckpt_path)

    # 小尺寸前向冒烟
    import time
    ir = torch.rand(1, 1, 64, 64)
    vis = torch.rand(1, 1, 64, 64)
    t0 = time.time()
    with torch.no_grad():
        out = model(ir, vis)
    print(f"\n前向冒烟 (64×64): 输出 {tuple(out.shape)}, 耗时 {(time.time()-t0)*1000:.0f}ms")
    print(f"输出值域: [{out.min():.4f}, {out.max():.4f}]")
    print("✅ LAMF_CPU 自测通过")
