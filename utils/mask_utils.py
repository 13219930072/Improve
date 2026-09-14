import torch
import kornia
import kornia.filters as filters

class GradientConflictMasker:
    """
    热显著性引导的软掩码生成器 (Thermal-Saliency Guided Soft Masker)
    核心创新：
    1. 热显著性加权：利用红外强度过滤背景可见光纹理的误杀。
    2. 软掩码平滑：通过高斯羽化消除绝对二值掩码带来的边缘振铃噪声。
    """
    def __init__(self, mask_ratio=0.15, smooth_mask=True):
        # 建议遮挡率下调至 15%，因为现在是精确狙击高亮热源边缘
        self.mask_ratio = mask_ratio
        # 默认开启掩码羽化，消除硬切断带来的高频噪点
        self.smooth_mask = smooth_mask

    def apply_mask(self, ir_tensor, vi_tensor):
        # 1. 提取空间梯度矩阵 [B, C, H, W]
        grad_ir = filters.SpatialGradient()(ir_tensor).abs().sum(dim=2)
        grad_vi = filters.SpatialGradient()(vi_tensor).abs().sum(dim=2)
        
        # 2. 计算原始空间梯度差异
        raw_conflict = torch.abs(grad_ir - grad_vi)
        
        # ==========================================
        # [核心突破] 3. 引入热显著性权重 (Thermal Saliency Guidance)
        # 物理意义：如果红外区域是冷的（像素值接近0），则忽略该处的边缘冲突
        # 保护可见光中健康的背景树木、建筑纹理不被无辜挖去！
        # ==========================================
        guided_conflict = raw_conflict * ir_tensor
        
        # 4. 动态阈值计算
        B, C, H, W = guided_conflict.shape
        conflict_flat = guided_conflict.view(B, -1)
        
        # 计算需要挖去的高危像素数量 k
        k = max(1, int(conflict_flat.shape[1] * self.mask_ratio))
        # 找到分界线阈值 (第 N-k 小的值)
        thresholds, _ = torch.kthvalue(conflict_flat, conflict_flat.shape[1] - k, dim=1)
        thresholds = thresholds.view(B, 1, 1, 1)
        
        # 5. 生成初始二值掩码 (冲突大于阈值设为0盲区，其余设为1保留区)
        mask = (guided_conflict <= thresholds).float()
        
        # ==========================================
        # [温和给药] 6. 软掩码平滑 (Soft Masking)
        # 用高斯模糊给掩码边缘加上“羽化”效果，避免网络输入产生断崖式的黑洞
        # ==========================================
        if self.smooth_mask:
            # 使用 5x5 高斯核进行轻微的边界平滑过渡
            mask = filters.gaussian_blur2d(mask, (5, 5), (1.5, 1.5))
            # 将极小的值归零，极大的值归一，保持核心区域的纯净度
            mask = torch.clamp(mask, 0.0, 1.0)
        
        # 7. 应用掩码：与原始图像物理相乘
        masked_ir = ir_tensor * mask
        masked_vi = vi_tensor * mask
        
        return masked_ir, masked_vi, mask
