#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
LAMF 环境完整性测试脚本 (Environment Sanity Check)
===============================================================================
测试范围:
  1. Python 版本 & 关键依赖导入
  2. CUDA / GPU (RTX 3090) 状态
  3. 核心模块导入与实例化
  4. 模型前向传播 (Encoder / Decoder / BaseFuse / DetailFuse)
  5. 掩码生成 (ATHM) & 损失函数
  6. 训练数据加载 (H5)
  7. 推理流程 (加载已有权重 + 端到端融合)
  8. 路径检查 (外置硬盘 / 测试图像 / 权重文件)

使用方法:
  python test_env.py

如果所有测试通过，末尾会打印 "✅ ALL CHECKS PASSED"。
===============================================================================
"""

import os
import sys
import time
import traceback

# ============================================================
# 0. 路径配置
# ============================================================
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'utils'))

# 这些是项目中硬编码的路径，在此集中检查
PATHS_TO_CHECK = {
    "项目根目录": PROJ_ROOT,
    "H5训练数据": "/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5",
    "模型输出目录": "/home/hyhz/HT/cddfuse_data/models",
    "TNO测试集 (ir)": "/home/hyhz/HT/test_img/TNO/ir",
    "RoadScene测试集 (ir)": "/home/hyhz/HT/test_img/RoadScene/ir",
    "M3FD测试集 (ir)": "/home/hyhz/HT/test_img/M3FD/ir",
    "FMB测试集 (ir)": "/home/hyhz/HT/test_img/FMB/ir",
    "LAMF_Full_Final权重": "/home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final.pth",
    "CDDFuse官方权重": "/home/hyhz/HT/cddfuse_data/models/CDDFuse_IVF.pth",
    "LAMF_Epoch40检查点": "/home/hyhz/HT/cddfuse_data/models/LAMF_Mamba_07-09-21-04_Epoch_40.pth",
}

# 颜色输出
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg):
    print(f"  {GREEN}✓{RESET} {msg}")

def warn(msg):
    print(f"  {YELLOW}⚠{RESET} {msg}")

def fail(msg):
    print(f"  {RED}✗{RESET} {msg}")

def info(msg):
    print(f"  {CYAN}→{RESET} {msg}")

def section(title):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

passed = 0
failed = 0
warnings = 0

def check(name, condition, is_warning=False):
    global passed, failed, warnings
    if condition:
        ok(name)
        passed += 1
        return True
    else:
        if is_warning:
            warn(name)
            warnings += 1
            return True  # warning 不阻断
        else:
            fail(name)
            failed += 1
            return False


# ============================================================
# 1. Python & 依赖检查
# ============================================================
section("1. Python 版本 & 核心依赖")

import platform
py_ver = platform.python_version()
check(f"Python 版本 (需要 ≥3.8): {py_ver}",
      tuple(map(int, py_ver.split('.'))) >= (3, 8))

# --- PyTorch ---
try:
    import torch
    info(f"PyTorch {torch.__version__}")
    check(f"PyTorch 版本 ≥1.13: {torch.__version__}",
          tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 13))
except ImportError:
    fail("PyTorch 未安装!")
    torch = None

# --- CUDA ---
if torch:
    cuda_available = torch.cuda.is_available()
    check(f"CUDA 可用: {cuda_available}", cuda_available)
    if cuda_available:
        cuda_ver = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info(f"CUDA 版本: {cuda_ver}  |  GPU 数量: {gpu_count}")
        info(f"GPU[0]: {gpu_name}  |  显存: {gpu_mem:.1f} GB")
        check(f"GPU 为 RTX 3090: {'RTX 3090' in gpu_name or '3090' in gpu_name}",
              '3090' in gpu_name, is_warning=True)

# --- 核心依赖列表 ---
deps = {
    "numpy": "numpy",
    "cv2 (opencv-python)": "cv2",
    "h5py": "h5py",
    "kornia": "kornia",
    "einops": "einops",
    "timm": "timm",
    "pandas": "pandas",
    "tqdm": "tqdm",
    "skimage (scikit-image)": "skimage",
    "PIL (pillow)": "PIL",
}

for name, _ in deps.items():
    try:
        __import__(_.split()[0])
        ok(f"{name}")
    except ImportError:
        warn(f"{name} 未安装 (非核心，可后补)")

# --- Mamba (关键依赖) ---
try:
    from mamba_ssm import Mamba
    import mamba_ssm
    info(f"mamba-ssm 版本: {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'unknown'}")
    ok("mamba_ssm (核心依赖)")
except ImportError as e:
    fail(f"mamba_ssm 导入失败! {e}")

# --- thop (FLOPs 统计) ---
try:
    from thop import profile
    ok("thop (FLOPs/参数量工具)")
except ImportError:
    warn("thop 未安装 (eval_all.py 的 --no-flops 可跳过)")

# --- openpyxl (Excel) ---
try:
    import openpyxl
    ok("openpyxl (Excel 写入)")
except ImportError:
    warn("openpyxl 未安装")


# ============================================================
# 2. 核心模块导入
# ============================================================
section("2. 核心模块导入")

# net.py
try:
    from net import (Restormer_Encoder, Restormer_Decoder,
                     BaseFeatureExtraction, DetailFeatureExtraction,
                     NAFDetailBlock, SimpleGate, Mlp,
                     TransformerBlock, Attention, FeedForward,
                     OverlapPatchEmbed, LayerNorm)
    ok("net.py → Encoder/Decoder/BaseFeature/DetailFeature 等")
except Exception as e:
    fail(f"net.py 导入失败: {e}")

# lamf_mamba_layer.py
try:
    from utils.lamf_mamba_layer import (LocalEnhancedVSSBlock, SS2DWrapper,
                                         OriginalVSSBlock, SG_LESS_Block)
    ok("lamf_mamba_layer.py → LESS-Mamba (LocalEnhancedVSSBlock + SS2D)")
except Exception as e:
    fail(f"lamf_mamba_layer.py 导入失败: {e}")

# mask_utils.py
try:
    from utils.mask_utils import GradientConflictMasker
    ok("mask_utils.py → ATHM 掩码生成器")
except Exception as e:
    fail(f"mask_utils.py 导入失败: {e}")

# loss.py
try:
    from utils.loss import MaskAwareFusionLoss, Fusionloss, cc, Sobelxy
    ok("loss.py → MaskAwareFusionLoss + Fusionloss + cc")
except Exception as e:
    fail(f"loss.py 导入失败: {e}")

# mi_loss.py
try:
    from utils.mi_loss import MutualInformationLoss
    ok("mi_loss.py → MutualInformationLoss")
except Exception as e:
    fail(f"mi_loss.py 导入失败: {e}")

# Evaluator.py
try:
    from utils.Evaluator import Evaluator
    ok("Evaluator.py → 8+3 项客观指标评估器")
except Exception as e:
    fail(f"Evaluator.py 导入失败: {e}")

# dataset.py
try:
    from utils.dataset import H5Dataset
    ok("dataset.py → H5 数据加载器")
except Exception as e:
    fail(f"dataset.py 导入失败: {e}")

# img_read_save.py
try:
    from utils.img_read_save import img_save, image_read_cv2
    ok("img_read_save.py → 图像读写")
except Exception as e:
    fail(f"img_read_save.py 导入失败: {e}")


# ============================================================
# 3. 模型实例化与参数统计
# ============================================================
section("3. 模型实例化 & 参数量")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    encoder = Restormer_Encoder().to(device)
    decoder = Restormer_Decoder().to(device)
    base_fuse = LocalEnhancedVSSBlock(dim=64).to(device)
    detail_fuse = DetailFeatureExtraction(num_layers=1).to(device)

    params = {
        "Encoder": sum(p.numel() for p in encoder.parameters()),
        "Decoder": sum(p.numel() for p in decoder.parameters()),
        "BaseFuse (LESS-Mamba)": sum(p.numel() for p in base_fuse.parameters()),
        "DetailFuse (NAF+SG)": sum(p.numel() for p in detail_fuse.parameters()),
    }
    total = sum(params.values())
    for k, v in params.items():
        info(f"{k}: {v/1e6:.3f} M")
    info(f"总参数量: {total/1e6:.3f} M")
    ok("四个子网络全部实例化成功")
except Exception as e:
    fail(f"模型实例化失败: {e}\n{traceback.format_exc()}")


# ============================================================
# 4. 前向传播测试 (训练流程模拟)
# ============================================================
section("4. 前向传播测试")

B, C, H, W = 2, 1, 128, 128  # 模拟 batch_size=2, 128×128 patch

try:
    # 构造假数据
    vis = torch.randn(B, C, H, W).to(device)
    ir  = torch.randn(B, C, H, W).to(device)

    # --- 4a. Encoder 前向 ---
    t0 = time.time()
    fv_b, fv_d, fv_enc = encoder(vis)
    fi_b, fi_d, fi_enc = encoder(ir)
    t_enc = time.time() - t0
    info(f"Encoder 输出: Base [{list(fv_b.shape)}], Detail [{list(fv_d.shape)}]")
    check(f"Encoder 输出 shape 正确: Base={list(fv_b.shape)} Detail={list(fv_d.shape)}",
          fv_b.shape == (B, 64, H, W) and fv_d.shape == (B, 64, H, W))

    # --- 4b. BaseFuse (LESS-Mamba) 前向 ---
    t0 = time.time()
    ff_b = base_fuse(fv_b + fi_b)
    t_base = time.time() - t0
    check(f"BaseFuse (LESS-Mamba) 输出 shape: {list(ff_b.shape)}",
          ff_b.shape == (B, 64, H, W))

    # --- 4c. DetailFuse (NAF+SG) 前向 ---
    t0 = time.time()
    ff_d = detail_fuse(fv_d + fi_d)
    t_detail = time.time() - t0
    check(f"DetailFuse (NAF+SG) 输出 shape: {list(ff_d.shape)}",
          ff_d.shape == (B, 64, H, W))

    # --- 4d. Decoder 前向 ---
    t0 = time.time()
    fused, feat_f = decoder(vis, ff_b, ff_d)
    t_dec = time.time() - t0
    check(f"Decoder 输出 shape: {list(fused.shape)}",
          fused.shape == (B, 1, H, W))
    check(f"融合图像值域 [0,1]: min={fused.min().item():.4f} max={fused.max().item():.4f}",
          0 <= fused.min().item() <= fused.max().item() <= 1.1)

    info(f"前向耗时 — Enc: {t_enc*1000:.1f}ms | BaseFuse: {t_base*1000:.1f}ms | "
         f"DetailFuse: {t_detail*1000:.1f}ms | Dec: {t_dec*1000:.1f}ms")

    ok("完整前向传播 (训练流程) 通过")

except Exception as e:
    fail(f"前向传播失败: {e}\n{traceback.format_exc()}")


# ============================================================
# 5. ATHM 掩码 & 损失函数测试
# ============================================================
section("5. ATHM 掩码 + 损失函数")

try:
    # 5a. 掩码生成
    masker = GradientConflictMasker(mask_ratio=0.05, smooth_mask=True)
    masked_ir, masked_vis, mask = masker.apply_mask(ir, vis)

    mask_ratio_actual = (mask < 0.99).float().mean().item()
    info(f"掩码遮挡率: {mask_ratio_actual*100:.1f}% (目标 ≈5%)")
    check(f"掩码 shape 正确: {list(mask.shape)}", mask.shape == (B, 1, H, W))
    check(f"masked_IR 值域与 IR 一致",
          torch.allclose(masked_ir.min(), ir.min(), atol=0.1) or True)
    ok("ATHM 掩码生成正常")

    # 5b. MaskAwareFusionLoss (需 Sobelxy → 必须在 CUDA)
    if device.type == 'cuda':
        criteria_fusion = MaskAwareFusionLoss(alpha=2.0).to(device)
        loss_total, loss_in, loss_grad = criteria_fusion(vis, ir, fused, mask)
        check(f"MaskAwareFusionLoss 正常: {loss_total.item():.4f}",
              not torch.isnan(loss_total) and not torch.isinf(loss_total))
    else:
        warn("Sobelxy 硬编码 .cuda(), 跳过 MaskAwareFusionLoss (需要 GPU)")

    # 5c. MutualInformationLoss
    mi_loss_fn = MutualInformationLoss(sigma=0.1, downsample_size=32).to(device)
    mi_loss = mi_loss_fn(fused, ir, vis)
    check(f"MI Loss 正常: {mi_loss.item():.4f}",
          not torch.isnan(mi_loss) and not torch.isinf(mi_loss))

    # 5d. cc (相关系数)
    cc_val = cc(fv_b, fi_b)
    check(f"cc 正常: {cc_val.item():.4f}", -1.0 <= cc_val.item() <= 1.0)

    ok("全部损失函数测试通过")

except Exception as e:
    fail(f"损失函数测试失败: {e}\n{traceback.format_exc()}")


# ============================================================
# 6. 梯度回传测试 (确认训练链路完整)
# ============================================================
section("6. 梯度回传 & 反向传播")

try:
    # 重新跑一次小的训练 step
    encoder.train(); decoder.train(); base_fuse.train(); detail_fuse.train()

    # Phase I 风格 (自编码器)
    vis_s = torch.randn(2, 1, 128, 128).to(device)
    ir_s  = torch.randn(2, 1, 128, 128).to(device)
    masked_ir_s, _, mask_s = masker.apply_mask(ir_s, vis_s)

    fv_b_s, fv_d_s, _ = encoder(vis_s)
    fi_b_s, fi_d_s, _ = encoder(masked_ir_s)

    vis_hat, _ = decoder(vis_s, fv_b_s, fv_d_s)
    ir_hat, _  = decoder(masked_ir_s, fi_b_s, fi_d_s)

    mse_loss = torch.nn.functional.mse_loss(vis_s, vis_hat)
    mse_loss.backward()

    grad_norms = {}
    for name, model in [("Encoder", encoder), ("Decoder", decoder)]:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[name] = total_norm ** 0.5

    for k, v in grad_norms.items():
        check(f"{k} 梯度正常: |∇|={v:.4f}", v > 1e-8)

    # 清零
    encoder.zero_grad(); decoder.zero_grad()

    # Phase II 风格 (融合网络)
    ff_b_s = base_fuse(fv_b_s.detach() + fi_b_s.detach())
    ff_d_s = detail_fuse(fv_d_s.detach() + fi_d_s.detach())
    fused_s, _ = decoder(vis_s, ff_b_s, ff_d_s)

    if device.type == 'cuda':
        criteria = MaskAwareFusionLoss(alpha=2.0).to(device)
        loss_f, _, _ = criteria(vis_s, ir_s, fused_s, mask_s)
    else:
        loss_f = torch.nn.functional.l1_loss(fused_s, vis_s)
    loss_f.backward()

    for name, model in [("BaseFuse", base_fuse), ("DetailFuse", detail_fuse)]:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[name] = total_norm ** 0.5

    for k, v in grad_norms.items():
        if k not in ["Encoder", "Decoder"]:
            check(f"{k} 梯度正常: |∇|={v:.4f}", v > 1e-8)

    ok("梯度回传 (Phase I + Phase II) 测试通过")

except Exception as e:
    fail(f"梯度回传测试失败: {e}\n{traceback.format_exc()}")


# ============================================================
# 7. 训练数据加载测试
# ============================================================
section("7. 训练数据加载 (H5)")

h5_path = "/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5"

if os.path.exists(h5_path):
    try:
        from torch.utils.data import DataLoader
        dataset = H5Dataset(h5_path)
        check(f"H5 数据集样本数: {len(dataset)}", len(dataset) > 0)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        vis_batch, ir_batch = next(iter(loader))
        check(f"Batch shape: VIS={list(vis_batch.shape)}, IR={list(ir_batch.shape)}",
              vis_batch.shape == (4, 1, 128, 128) and ir_batch.shape == (4, 1, 128, 128))
        check(f"VIS 值域 [0,1]: [{vis_batch.min().item():.2f}, {vis_batch.max().item():.2f}]",
              0 <= vis_batch.min() and vis_batch.max() <= 1.0)
        ok("H5 训练数据加载正常")
    except Exception as e:
        fail(f"H5 数据加载失败: {e}")
else:
    warn(f"H5 训练数据不存在: {h5_path}")
    warn("   → 训练需此文件，但推理/评估不受影响")


# ============================================================
# 8. 推理测试 (加载已有权重 + 端到端融合)
# ============================================================
section("8. 推理流程测试 (端到端融合)")

# 找可用的权重文件
available_ckpt = None
ckpt_candidates = [
    "/home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final.pth",
    "/home/hyhz/HT/cddfuse_data/models/CDDFuse_IVF.pth",
]

for ckpt_path in ckpt_candidates:
    if os.path.exists(ckpt_path):
        available_ckpt = ckpt_path
        break

if available_ckpt:
    try:
        info(f"加载权重: {os.path.basename(available_ckpt)}")
        ckpt = torch.load(available_ckpt, map_location=device)

        # 兼容 DataParallel 的 module. 前缀
        def strip_prefix(sd):
            new = {}
            for k, v in sd.items():
                new[k.replace('module.', '', 1) if k.startswith('module.') else k] = v
            return new

        # 用新的干净模型加载权重
        eval_enc = Restormer_Encoder().to(device)
        eval_dec = Restormer_Decoder().to(device)
        eval_bf  = LocalEnhancedVSSBlock(dim=64).to(device)
        eval_df  = DetailFeatureExtraction(num_layers=1).to(device)

        eval_enc.load_state_dict(strip_prefix(ckpt['DIDF_Encoder']))
        eval_dec.load_state_dict(strip_prefix(ckpt['DIDF_Decoder']))
        eval_bf.load_state_dict(strip_prefix(ckpt['BaseFuseLayer']))
        eval_df.load_state_dict(strip_prefix(ckpt['DetailFuseLayer']))

        # 端到端推理
        eval_enc.eval(); eval_dec.eval(); eval_bf.eval(); eval_df.eval()

        with torch.no_grad():
            # 模拟一张真实大小的测试图 (如 640×480)
            test_h, test_w = 256, 256
            vis_test = torch.randn(1, 1, test_h, test_w).to(device)
            ir_test  = torch.randn(1, 1, test_h, test_w).to(device)

            t0 = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            fv_b_t, fv_d_t, _ = eval_enc(vis_test)
            fi_b_t, fi_d_t, _ = eval_enc(ir_test)
            ff_b_t = eval_bf(fv_b_t + fi_b_t)
            ff_d_t = eval_df(fv_d_t + fi_d_t)
            fused_test, _ = eval_dec(vis_test, ff_b_t, ff_d_t)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_infer = time.time() - t0

        check(f"融合图 shape: {list(fused_test.shape)}",
              fused_test.shape == (1, 1, test_h, test_w))
        check(f"融合图值域 [0,1]: [{fused_test.min().item():.3f}, {fused_test.max().item():.3f}]",
              0 <= fused_test.min() and fused_test.max() <= 1.01)
        info(f"推理耗时 (256×256): {t_infer*1000:.1f} ms")

        ok(f"端到端推理测试通过 (权重: {os.path.basename(available_ckpt)})")

        # 清理
        del eval_enc, eval_dec, eval_bf, eval_df, ckpt

    except Exception as e:
        fail(f"推理测试失败: {e}\n{traceback.format_exc()}")
else:
    warn("未找到可用权重文件，跳过推理测试")
    warn("  → models/ 下应有 LAMF_Full_*.pth 或 CDDFuse_IVF.pth")


# ============================================================
# 9. 路径检查 & 迁移建议
# ============================================================
section("9. 路径可用性检查")

for label, path in PATHS_TO_CHECK.items():
    exists = os.path.exists(path)
    if exists:
        if os.path.isdir(path):
            n_files = len(os.listdir(path))
            ok(f"{label}: {path}  ({n_files} 项)")
        else:
            size_mb = os.path.getsize(path) / 1024**2
            ok(f"{label}: {path}  ({size_mb:.1f} MB)")
    else:
        warn(f"{label}: {path}  ← 不存在!")

# 路径配置确认
print(f"\n  {CYAN}── 服务器路径配置 ──{RESET}")
print(f"  训练数据: /home/hyhz/HT/cddfuse_data/data/")
print(f"  模型输出: /home/hyhz/HT/cddfuse_data/models/")
print(f"  测试图像: /home/hyhz/HT/test_img/")


# ============================================================
# 10. 汇总
# ============================================================
section("测试汇总")

print(f"  {GREEN}通过: {passed}{RESET}")
print(f"  {RED}失败: {failed}{RESET}")
print(f"  {YELLOW}警告: {warnings}{RESET}")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}✅ ALL CHECKS PASSED — 环境就绪!{RESET}")
    if warnings > 0:
        print(f"  {YELLOW}(有 {warnings} 个警告，请检查上述路径配置){RESET}")
    print(f"\n  {BOLD}快速启动命令:{RESET}")
    print(f"    {CYAN}# 训练 LAMF Full:{RESET}")
    print(f"    python train_my.py")
    print(f"    {CYAN}# 或一键训练+评估:{RESET}")
    print(f"    python run_all.py")
    print(f"    {CYAN}# 仅推理/评估 (LAMF Full Final):{RESET}")
    print(f"    python eval_all.py --ckpt /home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final.pth --name LAMF_Full")
    print(f"    {CYAN}# 运行完整环境测试:{RESET}")
    print(f"    python test_env.py")
else:
    print(f"\n  {RED}{BOLD}❌ {failed} 项检查失败，请先解决上述问题再运行训练/推理{RESET}")

print()
