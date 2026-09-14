import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 你的创新点：掩码感知解耦融合损失 (Mask-Aware Decoupled Fusion Loss)
# ==========================================
class MaskAwareFusionLoss(nn.Module):
    def __init__(self, alpha=2.0):
        """
        :param alpha: 对 25% 掩码盲区的惩罚权重放大系数。
                      网络在盲区没有输入信息，我们用更大的权重强迫它进行跨域结构推理。
        """
        super(MaskAwareFusionLoss, self).__init__()
        self.sobelconv = Sobelxy()
        self.alpha = alpha

    def forward(self, image_vis, image_ir, generate_img, mask):
        # 提取单通道进行梯度和强度计算 (适配 CDDFuse 默认单通道设定)
        image_y = image_vis[:, :1, :, :]
        image_ir_y = image_ir[:, :1, :, :]
        generate_img_y = generate_img[:, :1, :, :]

        # 保证 mask 维度一致 [B, 1, H, W]
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # 1. 基础强度损失 (未 reduction，保留空间维度以便施加 mask)
        x_in_max = torch.max(image_y, image_ir_y)
        loss_in_map = F.l1_loss(x_in_max, generate_img_y, reduction='none')

        # 2. 基础梯度损失 (保留空间维度)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir_y)
        generate_img_grad = self.sobelconv(generate_img_y)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad_map = F.l1_loss(x_grad_joint, generate_img_grad, reduction='none')

        # ==========================================
        # 3. 核心创新：掩码感知解耦 (Mask-Aware Decoupling)
        # mask == 1: 75% 的平滑保留区
        # mask == 0: 25% 的冲突遮挡区 (盲区)
        # ==========================================
        mask_inv = 1.0 - mask  # 反转掩码，定位到盲区

        # 对 75% 的正常区域，施加标准的 L1 约束
        loss_in_normal = (loss_in_map * mask).mean()
        loss_grad_normal = (loss_grad_map * mask).mean()

        # 对 25% 的盲区，施加 alpha 倍的严格结构约束 (因为这里最容易产生重影)
        loss_in_blind = (loss_in_map * mask_inv).mean()
        loss_grad_blind = (loss_grad_map * mask_inv).mean()

        # 加权融合
        loss_in_total = loss_in_normal + self.alpha * loss_in_blind
        loss_grad_total = loss_grad_normal + self.alpha * loss_grad_blind

        loss_total = loss_in_total + 10 * loss_grad_total
        return loss_total, loss_in_total, loss_grad_total

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()