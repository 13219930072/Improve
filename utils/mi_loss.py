import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInformationLoss(nn.Module):
    """ 最大化融合图与红外、可见光的互信息 (优化防OOM版) """
    def __init__(self, sigma=0.1, reduction='mean', downsample_size=32):
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction
        # 核心修复：将 128x128 压缩到 32x32 获取全局统计分布
        self.downsample_size = downsample_size 

    def _gaussian_kernel(self, x, y, sigma):
        # x, y: [N, D]
        dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    def _entropy(self, x, sigma):
        N = x.shape[0]
        K = self._gaussian_kernel(x, x, sigma)          # [N, N]
        p_x = K.mean(dim=1)                             # [N]
        log_p = torch.log(p_x + 1e-8)
        entropy = -log_p.mean()
        return entropy

    def _joint_entropy(self, x, y, sigma):
        xy = torch.cat([x, y], dim=1)                   # [N, 2D]
        K = self._gaussian_kernel(xy, xy, sigma)
        p_xy = K.mean(dim=1)
        log_p = torch.log(p_xy + 1e-8)
        joint_entropy = -log_p.mean()
        return joint_entropy

    def forward(self, fused, source_ir, source_vis):
        """
        fused: [B, C, H, W]
        """
        # ==========================================
        # [防爆显存机制] 降采样
        # 统计互信息不需要全分辨率，只需提取全局密度
        # ==========================================
        if self.downsample_size:
            fused = F.adaptive_avg_pool2d(fused, self.downsample_size)
            source_ir = F.adaptive_avg_pool2d(source_ir, self.downsample_size)
            source_vis = F.adaptive_avg_pool2d(source_vis, self.downsample_size)

        B, C, H, W = fused.shape
        
        # 展平为 [N, C]
        fused_flat = fused.permute(0,2,3,1).reshape(-1, C)
        ir_flat = source_ir.permute(0,2,3,1).reshape(-1, C)
        vis_flat = source_vis.permute(0,2,3,1).reshape(-1, C)

        H_fused = self._entropy(fused_flat, self.sigma)
        H_ir = self._entropy(ir_flat, self.sigma)
        H_vis = self._entropy(vis_flat, self.sigma)

        H_fused_ir = self._joint_entropy(fused_flat, ir_flat, self.sigma)
        H_fused_vis = self._joint_entropy(fused_flat, vis_flat, self.sigma)

        mi_ir = H_fused + H_ir - H_fused_ir
        mi_vis = H_fused + H_vis - H_fused_vis

        loss = - (mi_ir + mi_vis)      # 最大化MI -> 最小化 -MI
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss