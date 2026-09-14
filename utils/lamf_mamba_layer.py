import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mamba_ssm import Mamba  # 确保你已经安装了本地的 mamba_ssm=1.0.1 库

class SS2DWrapper(nn.Module):
    """
    基于 mamba_ssm 1.0.1 的四向扫描封装
    """
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=16,      
            d_conv=4,        
            expand=2,        
        )

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = x.view(B, H * W, C) 
        
        # 四向展开
        x_forward = x_flat
        x_backward = torch.flip(x_flat, dims=[1])
        x_trans = x.transpose(1, 2).contiguous().view(B, H * W, C)
        x_trans_back = torch.flip(x_trans, dims=[1])

        # Mamba 扫描
        y_forward = self.mamba(x_forward)
        y_backward = torch.flip(self.mamba(x_backward), dims=[1]) 
        
        y_trans_out = self.mamba(x_trans)
        y_trans = y_trans_out.view(B, W, H, C).transpose(1, 2).contiguous().view(B, H * W, C)
        
        y_trans_back_out = torch.flip(self.mamba(x_trans_back), dims=[1])
        y_trans_back = y_trans_back_out.view(B, W, H, C).transpose(1, 2).contiguous().view(B, H * W, C)

        y = (y_forward + y_backward + y_trans + y_trans_back) / 4.0
        return y.view(B, H, W, C)

class LocalEnhancedVSSBlock(nn.Module):
    """
    LAMF 专属: LESS-Mamba + 内部 SimpleGate 重构
    """
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.project_in = nn.Linear(dim, dim * 2)
        
        # 局部高频增强分支
        self.dwconv = nn.Conv2d(
            in_channels=dim, out_channels=dim, 
            kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.ss2d = SS2DWrapper(d_model=dim)
        self.project_out = nn.Linear(dim, dim)

    @autocast() 
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        shortcut = x

        x = self.ln(x)
        x_proj = self.project_in(x) 
        
        # 【创新点注入】：斩断 SiLU/GELU，直接在这里切分通道
        x1, x2 = x_proj.chunk(2, dim=-1) 

        # --- 局部增强 + Mamba 全局扫描 ---
        x1 = x1.permute(0, 3, 1, 2).contiguous() 
        x1 = self.dwconv(x1) 
        x1 = x1.permute(0, 2, 3, 1).contiguous() 
        
        x1 = self.ss2d(x1) 

        # 【创新点注入】：内部 SimpleGate 门控交互 (零参数)
        out = x1 * x2
        
        out = self.project_out(out)
        out = out + shortcut

        return out.permute(0, 3, 1, 2).contiguous()


class OriginalVSSBlock(nn.Module):
    """
    原生 MambaIR 模块 (照搬原始设计，内部使用 SiLU 门控)

    用于消融实验 D 和 D_plus，与 LocalEnhancedVSSBlock 的唯一区别：
    - 门控方式: SiLU (F.silu(x1) * x2) 而非 SimpleGate (x1 * x2)
    - DWConv 通道数: dim (而非 dim*2 的扩展)

    数据流:
        x → LayerNorm → Linear(64→128) → chunk(2)
        → x1: DWConv → SS2D → SiLU
        → x2: bypass (门控信号)
        → out = silu(x1) * x2
        → project_out + residual
    """
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.project_in = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv2d(
            in_channels=dim, out_channels=dim,
            kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.ss2d = SS2DWrapper(d_model=dim)
        self.project_out = nn.Linear(dim, dim)

    @autocast()
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        shortcut = x

        x = self.ln(x)
        x_proj = self.project_in(x)
        x1, x2 = x_proj.chunk(2, dim=-1)

        # 局部增强 + Mamba 全局扫描
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.dwconv(x1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.ss2d(x1)

        # 【关键区别】：原生 SiLU 门控，而非 SimpleGate
        out = F.silu(x1) * x2

        out = self.project_out(out)
        out = out + shortcut

        return out.permute(0, 3, 1, 2).contiguous()


# 别名：方便评估脚本导入 (SG = SimpleGate)
SG_LESS_Block = LocalEnhancedVSSBlock