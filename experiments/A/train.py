# -*- coding: utf-8 -*-
"""
===========================================================================
消融实验 A: CDDFuse 基线 (Transformer + SimpleDetailFusion)
===========================================================================
架构:
  Base 融合:   BaseFeatureExtraction (Transformer Attention, CDDFuse 原版)
  Detail 融合: SimpleDetailFusion (3层卷积 + LeakyReLU, 模拟 CDDFuse 原版)
  损失函数:    Fusionloss only (intensity + gradient + decomp)

消融链位置:
  A  (Transformer + SimpleDetail)          ← 本实验 (基线)
  C  (Transformer + NAFDetailBlock SG)     ← +SimpleGate
  D  (OriginalVSSBlock SiLU + SimpleDetail) ← +原生 MambaIR
  D+ (OriginalVSSBlock SiLU + NAFDetail SG) ← bridge
  E  (LocalEnhancedVSSBlock SG + NAFDetail) ← LAMF final

用法:
  python train.py
===========================================================================
"""
import os
import sys
import time
import datetime

# ---- 项目根目录 ----
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, 'utils'))

# ---- 本实验专属路径 ----
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- 公共数据路径 ----
H5_PATH = "/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5"

# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import kornia

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc

# 导入实验公共模块中的极简 Detail 融合层
sys.path.insert(0, os.path.join(PROJ_ROOT, 'experiments'))
from common import SimpleDetailFusion

# -----------------------------------------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 损失函数: 仅使用原版 Fusionloss
criteria_fusion = Fusionloss().to(device)

# 训练超参数
num_epochs = 120
epoch_gap = 40
lr = 1e-4
weight_decay = 0
batch_size = 8
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# 损失系数
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

# 模型实例化: Transformer(BaseFeatureExtraction) + SimpleDetailFusion
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
# A 的核心: Base = Transformer, Detail = 极简卷积融合
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(SimpleDetailFusion(dim=64)).to(device)

# 优化器
optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(window_size=11, max_val=1.0, reduction='mean')

# AMP 混合精度
scaler = GradScaler()

# 数据加载
trainloader = DataLoader(
    H5Dataset(H5_PATH),
    batch_size=batch_size, shuffle=True, num_workers=0
)
loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

print(f"实验: A (CDDFuse 基线 — Transformer + SimpleDetailFusion)")
print(f"训练数据: {H5_PATH}")
print(f"模型保存: {CKPT_DIR}")
print(f"设备: {device}")
print("=" * 60)

# =============================================================================
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        input_VIS = data_VIS
        input_IR = data_IR

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        with autocast():
            if epoch < epoch_gap:  # Phase I: 自编码器预训练
                feature_V_B, feature_V_D, _ = DIDF_Encoder(input_VIS)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(input_IR)
                data_VIS_hat, _ = DIDF_Decoder(input_VIS, feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(input_IR, feature_I_B, feature_I_D)

                cc_loss_B = cc(feature_V_B.float(), feature_I_B.float())
                cc_loss_D = cc(feature_V_D.float(), feature_I_D.float())
                mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
                mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

                Gradient_loss = L1Loss(
                    kornia.filters.SpatialGradient()(data_VIS),
                    kornia.filters.SpatialGradient()(data_VIS_hat)
                )

                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                loss = (coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I +
                        coeff_decomp * loss_decomp + coeff_tv * Gradient_loss)

            else:  # Phase II: 融合网络训练
                feature_V_B, feature_V_D, feature_V = DIDF_Encoder(input_VIS)
                feature_I_B, feature_I_D, feature_I = DIDF_Encoder(input_IR)

                # A 核心: Transformer 融合 Base, 极简卷积融合 Detail
                feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

                data_Fuse, feature_F = DIDF_Decoder(input_VIS, feature_F_B, feature_F_D)

                cc_loss_B = cc(feature_V_B.float(), feature_I_B.float())
                cc_loss_D = cc(feature_V_D.float(), feature_I_D.float())
                loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

                fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

                loss = fusionloss + coeff_decomp * loss_decomp

        # AMP: 梯度缩放反向传播
        scaler.scale(loss).backward()

        # AMP: 梯度裁剪前必须先 unscale
        scaler.unscale_(optimizer1)
        scaler.unscale_(optimizer2)
        nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        if not epoch < epoch_gap:
            scaler.unscale_(optimizer3)
            scaler.unscale_(optimizer4)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

        # AMP: scaler.step() + update() 替代 optimizer.step()
        scaler.step(optimizer1)
        scaler.step(optimizer2)
        if not epoch < epoch_gap:
            scaler.step(optimizer3)
            scaler.step(optimizer4)
        scaler.update()

        # 打印日志
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (epoch, num_epochs, i, len(loader['train']), loss.item(), time_left)
        )

    # 调整学习率
    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    for opt in [optimizer1, optimizer2, optimizer3, optimizer4]:
        if opt.param_groups[0]['lr'] <= 1e-6:
            opt.param_groups[0]['lr'] = 1e-6

    if (epoch + 1) == 40:
        torch.save({
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }, os.path.join(CKPT_DIR, f"A_baseline_{timestamp}_Epoch_40.pth"))
        print(f"\n  [保存] Epoch 40 checkpoint → {CKPT_DIR}")

# 最终保存
final_path = os.path.join(CKPT_DIR, f"A_baseline_Final_{timestamp}.pth")
torch.save({
    'DIDF_Encoder': DIDF_Encoder.state_dict(),
    'DIDF_Decoder': DIDF_Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}, final_path)
print(f"\n✓ 训练完成! 最终权重: {final_path}")
print(f"  评估命令: python ../../eval_all.py --ckpt {final_path} --name A_baseline --variant A")
