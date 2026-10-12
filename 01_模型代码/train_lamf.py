# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia

# 导入网络模块 (注意：移除了原版的 BaseFeatureExtraction)
from net import Restormer_Encoder, Restormer_Decoder, DetailFeatureExtraction

# 导入工具、损失与 Mamba (根据截图结构，已全部统一到 utils 下)
from utils.dataset import H5Dataset
from utils.lamf_mamba_layer import LocalEnhancedVSSBlock
from utils.mask_utils import GradientConflictMasker
from utils.loss import MaskAwareFusionLoss, cc
from utils.mi_loss import MutualInformationLoss

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# [创新点] 实例化掩码生成器与新损失函数
masker = GradientConflictMasker(mask_ratio=0.05, smooth_mask=True)
criteria_fusion = MaskAwareFusionLoss(alpha=2.0).to(device)
mi_loss_fn = MutualInformationLoss(sigma=0.1).to(device)
lambda_mi = 0.3

# 严格对齐原版超参数
num_epochs = 120 
epoch_gap = 40   
lr = 1e-4
weight_decay = 0
batch_size = 8
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# 严格对齐原版损失系数
coeff_mse_loss_VF = 1. 
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      
coeff_tv = 5.

# 模型实例化 (BaseFuseLayer 被替换为 Mamba)
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(LocalEnhancedVSSBlock(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# 优化器与调度器
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
Loss_ssim = kornia.losses.SSIMLoss(window_size=11,max_val=1.0,reduction='mean')

# 数据加载
trainloader = DataLoader(H5Dataset(r"/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size, shuffle=True, num_workers=0)
loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        
        # [创新点] 生成掩码并仅对红外图像施加遮挡
        masked_IR, _, mask = masker.apply_mask(data_IR, data_VIS)
        mask = mask.cuda()
        
        # 定义输入
        input_VIS = data_VIS
        input_IR = masked_IR

        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: # Phase I: Autoencoder 训练
            feature_V_B, feature_V_D, _ = DIDF_Encoder(input_VIS)
            feature_I_B, feature_I_D, _ = DIDF_Encoder(input_IR)
            data_VIS_hat, _ = DIDF_Decoder(input_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(input_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + \
                   coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
            
        else:  # Phase II: 满血融合网络训练
            feature_V_B, feature_V_D, feature_V = DIDF_Encoder(input_VIS)
            feature_I_B, feature_I_D, feature_I = DIDF_Encoder(input_IR)
            
            # [创新点] Mamba全局推理 + NAFDetail细节提取
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
            
            data_Fuse, feature_F = DIDF_Decoder(input_VIS, feature_F_B, feature_F_D)  

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            
            # [创新点] 掩码感知融合损失 + 互信息兜底
            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse, mask)
            mi_loss = mi_loss_fn(data_Fuse, data_IR, data_VIS)
            
            loss = fusionloss + coeff_decomp * loss_decomp + lambda_mi * mi_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            
            optimizer1.step()  
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

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
    
    # 周期性保存模型 (只保存第四十轮)
    if (epoch + 1) == 40:
        os.makedirs("/home/hyhz/HT/cddfuse_data/models", exist_ok=True)
        torch.save({
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }, os.path.join("/home/hyhz/HT/cddfuse_data/models", f"LAMF_Full_{timestamp}_Epoch_{epoch+1}.pth"))

# 最终保存
torch.save({
    'DIDF_Encoder': DIDF_Encoder.state_dict(),
    'DIDF_Decoder': DIDF_Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}, os.path.join("/home/hyhz/HT/cddfuse_data/models", f"LAMF_Full_Final_{timestamp}.pth"))