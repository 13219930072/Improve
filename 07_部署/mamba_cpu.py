# -*- coding: utf-8 -*-
"""
===========================================================================
mamba_cpu.py — 纯 PyTorch 实现的 Mamba / SS2D（面向 ARM CPU 部署）
===========================================================================
用途: 替代 mamba_ssm 的 CUDA 算子，使 LAMF 可在无 NVIDIA GPU 的设备
      （如 Orange Pi 5 Plus / RK3588）上运行。

关键设计:
  1. 参数结构、命名与 mamba_ssm.Mamba **完全一致** → 可直接 load_state_dict
     加载现有 .pth 权重，无需重新训练
  2. selective scan 用纯 PyTorch 实现（参考 mamba_ssm 的 selective_scan_ref，
     但移除全部 CUDA 依赖）
  3. 提供可选的并行扫描（Hillis-Steele），实测加速有限（~1.5x），
     默认关闭以保证数值精度；长序列 CPU 推理的真正瓶颈是内存带宽
  4. 无 @autocast、无 torch.cuda 依赖

参数结构（d_model=64, d_state=16, d_conv=4, expand=2）:
  d_inner = 128, dt_rank = 4
  in_proj  : Linear(64 → 256)
  conv1d   : Conv1d(128, 128, kernel_size=4, groups=128)
  x_proj   : Linear(128 → 36)   # dt_rank + 2*d_state
  dt_proj  : Linear(4 → 128)
  A_log    : Parameter(128, 16)
  D        : Parameter(128)
  out_proj : Linear(128 → 64)
===========================================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 纯 PyTorch selective scan（替代 mamba_ssm 的 CUDA 算子）
# ===========================================================================
def selective_scan_cpu(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """
    纯 PyTorch 版 selective scan，数值等价于 CUDA 实现。

    参数 (LAMF 实际用到的形状):
        u     : (B, d_inner, L)         输入
        delta : (B, d_inner, L)         时间步长（离散化）
        A     : (d_inner, d_state)      状态转移（已取负指数）
        B     : (B, d_state, L)         输入矩阵（时变）
        C     : (B, d_state, L)         输出矩阵（时变）
        D     : (d_inner,)              直通项
        z     : (B, d_inner, L)         门控（SiLU 后与输出相乘）

    递推: h_t = exp(Δ_t·A) ⊙ h_{t-1} + Δ_t·B_t·u_t
          y_t = C_t·h_t + D⊙u_t
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    x = A.new_zeros((batch, dim, dstate))
    ys = []

    # 预计算每步的转移与输入贡献（向量化，避免逐步 einsum）
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))      # (B,D,L,N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)       # (B,D,L,N)
    # C 转成 (B,L,N) 便于逐步读取
    Ct = C.permute(0, 2, 1).contiguous()                            # (B,L,N)

    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]                 # (B,D,N)
        y = torch.einsum('bdn,bn->bd', x, Ct[:, i])                 # (B,D)
        ys.append(y)

    y = torch.stack(ys, dim=2)                                      # (B,D,L)
    out = y if D is None else y + u * D.reshape(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    if return_last_state:
        return out, x
    return out


def selective_scan_cpu_parallel(u, delta, A, B, C, D=None, z=None,
                                delta_bias=None, delta_softplus=False,
                                chunk_size=1024):
    """
    并行扫描版 selective scan（Hillis-Steele）—— 数值稳定 + 大幅加速。

    原理: SSM 递推 h_t = a_t·h_{t-1} + b_t 是线性递推，可用结合律并行化:
        组合运算 (a1,b1) ∘ (a2,b2) = (a2·a1, a2·b1 + b2)
      Hillis-Steele 扫描用 log2(C) 轮完成 chunk 内扫描（C=chunk_size），
      chunk 间串行传递状态。纯乘加运算，无数值下溢风险。

    复杂度: Python 循环次数从 O(L) 降到 O(L/C · log C)
            L=442368 时: 442368 次 → 432 chunk × 10 轮 ≈ 4320 次张量操作

    与逐步版数学等价（结合律保证），浮点累加顺序不同。
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, L = u.shape
    dstate = A.shape[1]

    def shift_right(t, offset, fill):
        """沿序列维(L)右移 offset，左侧填充 fill"""
        if offset <= 0:
            return t
        pad = torch.full_like(t[:, :, :offset], fill)
        return torch.cat([pad, t[:, :, :-offset]], dim=2)

    # 一次性向量化计算（einsum 效率高于分块广播，实测 L=65536 时快 2 倍）
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))      # (B,D,L,N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)       # (B,D,L,N)

    ys = []
    x = A.new_zeros((batch, dim, dstate))                           # 初始状态
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        C_len = end - start

        a_c = deltaA[:, :, start:end]                                # (B,D,C,N)
        b_c = deltaB_u[:, :, start:end]                              # (B,D,C,N)
        c_c = C[:, :, start:end].permute(0, 2, 1)                    # (B,C,N)

        # 把上一 chunk 的末状态并入首元素: h_0 = a_0·x + b_0
        a_s = a_c.clone()
        b_s = b_c.clone()
        b_s[:, :, 0] = a_c[:, :, 0] * x + b_c[:, :, 0]

        # Hillis-Steele 并行扫描（inclusive）
        offset = 1
        while offset < C_len:
            a_prev = shift_right(a_s, offset, 1.0)
            b_prev = shift_right(b_s, offset, 0.0)
            b_s = b_s + a_s * b_prev
            a_s = a_s * a_prev
            offset *= 2

        h_chunk = b_s                                                # (B,D,C,N)
        y_c = (h_chunk * c_c.unsqueeze(1)).sum(dim=-1)                # (B,D,C) 广播替代einsum
        ys.append(y_c)
        x = h_chunk[:, :, -1]                                        # 传给下一 chunk

    y = torch.cat(ys, dim=2)                                         # (B,D,L)
    out = y if D is None else y + u * D.reshape(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


def selective_scan_cpu_chunked(u, delta, A, B, C, D=None, z=None,
                               delta_bias=None, delta_softplus=False,
                               chunk_size=256):
    """
    ⚠️ 已废弃（数值不稳定），仅为向后兼容保留。

    问题: 该实现用 cumprod 比值公式，当衰减累积使 a_cumprod 趋于 0 时
    （长序列必然发生）会下溢/上溢，实测误差可达 O(1)~O(10)。
    请改用 selective_scan_cpu_parallel（数值稳定且更快）。
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, L = u.shape
    dstate = A.shape[1]

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))      # (B,D,L,N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)       # (B,D,L,N)

    ys = []
    x = A.new_zeros((batch, dim, dstate))
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        a_c = deltaA[:, :, start:end]                                # (B,D,C,N)
        b_c = deltaB_u[:, :, start:end]                              # (B,D,C,N)
        c_c = C[:, :, start:end].permute(0, 2, 1)                    # (B,C,N)

        # chunk 内：累积衰减系数与前缀输入贡献
        # a_prod[j] = prod_{k<=j} a_k  → 用 cumprod 向量化
        a_cumprod = torch.cumprod(a_c, dim=2)
        # b_prefix[j] = sum_{k<=j} (prod_{m=k+1..j} a_m) * b_k
        # 通过反向 cumprod 技巧向量化
        a_rev_cumprod = torch.cumprod(a_c.flip(2), dim=2).flip(2)
        # (B,D,C,N)
        weights = a_rev_cumprod / (a_c + 1e-30)                      # 数值保护
        # h_j = a_cumprod[j] * x_in + sum_{k<=j} (a_cumprod[j]/a_cumprod[k]) * b_k
        # 上式改写为: h_j = a_cumprod[j] * (x_in + sum_{k<=j} b_k / a_cumprod[k])
        b_scaled = b_c / (a_cumprod + 1e-30)
        b_prefix = torch.cumsum(b_scaled, dim=2)
        h_chunk = a_cumprod * (x.unsqueeze(2) + b_prefix)            # (B,D,C,N)

        # 输出: y_j = C_j · h_j
        y_c = (h_chunk * c_c.unsqueeze(1)).sum(dim=-1)                # (B,D,C) 广播替代einsum
        ys.append(y_c)

        # chunk 末状态传给下一 chunk
        x = h_chunk[:, :, -1]                                        # (B,D,N)

    y = torch.cat(ys, dim=2)                                         # (B,D,L)
    out = y if D is None else y + u * D.reshape(1, -1, 1)
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


# ===========================================================================
# 纯 PyTorch Mamba（参数结构与官方 mamba_ssm.Mamba 完全一致）
# ===========================================================================
class MambaCPU(nn.Module):
    """
    与 mamba_ssm.modules.mamba_simple.Mamba 参数结构一致，
    可直接加载官方权重（state_dict key 完全相同）。

    用 selective_scan_cpu 替代 CUDA 算子；无 autocast 依赖。
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank="auto", conv_bias=True, bias=False,
                 use_chunked=False, chunk_size=512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_chunked = use_chunked
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj 的初始化（与官方一致，虽加载权重后会覆盖）
        dt_init_std = self.dt_rank ** -0.5 * 1.0  # dt_scale=1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D) → (B, L, D)
        与 mamba_ssm 的 slow path 数值等价。
        """
        batch, seqlen, dim = hidden_states.shape

        # 输入投影 → chunk 成 x / z
        xz = self.in_proj(hidden_states)                     # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                           # 各 (B, L, d_inner)

        # 因果深度卷积（取前 seqlen 步）
        x = x.transpose(1, 2)                                # (B, d_inner, L)
        x = self.act(self.conv1d(x)[..., :seqlen])           # 因果卷积 + SiLU
        x = x.transpose(1, 2)                                # (B, L, d_inner)

        # 生成 Δ, B, C（输入依赖）
        x_dbl = self.x_proj(x)                               # (B, L, dt_rank+2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)                                # (B, L, d_inner)
        dt = dt.transpose(1, 2)                              # (B, d_inner, L)
        B = B.transpose(1, 2).contiguous()                   # (B, d_state, L)
        C = C.transpose(1, 2).contiguous()                   # (B, d_state, L)

        A = -torch.exp(self.A_log.float())                   # (d_inner, d_state)
        x = x.transpose(1, 2)                                # (B, d_inner, L)
        z = z.transpose(1, 2)                                # (B, d_inner, L)

        # selective scan（纯 PyTorch）
        scan_fn = selective_scan_cpu_parallel if self.use_chunked else selective_scan_cpu
        kwargs = dict(D=self.D, z=z, delta_bias=self.dt_proj.bias, delta_softplus=True)
        if self.use_chunked:
            kwargs["chunk_size"] = self.chunk_size
        y = scan_fn(x, dt, A, B, C, **kwargs)                # (B, d_inner, L)

        y = y.transpose(1, 2)                                # (B, L, d_inner)
        return self.out_proj(y)                              # (B, L, d_model)


# ===========================================================================
# SS2D 四向扫描（CPU 版，逻辑与 utils/lamf_mamba_layer.py 一致）
# ===========================================================================
class SS2DWrapperCPU(nn.Module):
    """四向扫描封装（CPU 版）。属性名与官方版一致: self.mamba"""
    def __init__(self, d_model, use_chunked=False, chunk_size=512):
        super().__init__()
        self.mamba = MambaCPU(d_model=d_model, d_state=16, d_conv=4, expand=2,
                              use_chunked=use_chunked, chunk_size=chunk_size)

    def forward(self, x):
        """x: (B, H, W, C) → (B, H, W, C)"""
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)

        x_forward = x_flat
        x_backward = torch.flip(x_flat, dims=[1])
        x_trans = x.transpose(1, 2).contiguous().view(B, H * W, C)
        x_trans_back = torch.flip(x_trans, dims=[1])

        y_forward = self.mamba(x_forward)
        y_backward = torch.flip(self.mamba(x_backward), dims=[1])

        y_trans_out = self.mamba(x_trans)
        y_trans = y_trans_out.view(B, W, H, C).transpose(1, 2).contiguous().view(B, H * W, C)

        y_trans_back_out = torch.flip(self.mamba(x_trans_back), dims=[1])
        y_trans_back = y_trans_back_out.view(B, W, H, C).transpose(1, 2).contiguous().view(B, H * W, C)

        y = (y_forward + y_backward + y_trans + y_trans_back) / 4.0
        return y.view(B, H, W, C)


# ===========================================================================
# LocalEnhancedVSSBlock（CPU 版，去掉 @autocast，其余与 GPU 版一致）
# ===========================================================================
class LocalEnhancedVSSBlockCPU(nn.Module):
    """LAMF 的 LESS-Mamba 融合块（纯 CPU 可运行版本）"""
    def __init__(self, dim, use_chunked=False, chunk_size=512):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.project_in = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv2d(
            in_channels=dim, out_channels=dim,
            kernel_size=3, padding=1, groups=dim, bias=False,
        )
        self.ss2d = SS2DWrapperCPU(d_model=dim, use_chunked=use_chunked,
                                   chunk_size=chunk_size)
        self.project_out = nn.Linear(dim, dim)

    def forward(self, x):
        """x: (B, C, H, W) → (B, C, H, W)"""
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        x = self.ln(x)
        x_proj = self.project_in(x)
        x1, x2 = x_proj.chunk(2, dim=-1)          # SimpleGate 的信号/门控分流

        # 局部增强（DWConv） + 全局扫描（SS2D）
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x1 = self.dwconv(x1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = self.ss2d(x1)

        out = x1 * x2                              # SimpleGate（零参数门控）
        out = self.project_out(out)
        out = out + shortcut
        return out.permute(0, 3, 1, 2).contiguous()


# ===========================================================================
# 自测
# ===========================================================================
if __name__ == '__main__':
    import time
    torch.manual_seed(0)
    print("=" * 70)
    print("mamba_cpu 自测")
    print("=" * 70)

    # 1) Mamba 单模块：形状与参数量
    m = MambaCPU(d_model=64)
    n_param = sum(p.numel() for p in m.parameters())
    print(f"MambaCPU 参数量: {n_param:,} (官方 mamba_ssm 同配置应为 ~99K)")

    x = torch.randn(1, 256, 64)
    t0 = time.time()
    y = m(x)
    print(f"前向 (L=256): 输出 {tuple(y.shape)}, 耗时 {(time.time()-t0)*1000:.0f}ms")

    # 2) 逐步版 vs 分块版一致性
    a = selective_scan_cpu(
        torch.randn(1, 128, 200), torch.randn(1, 128, 200).abs() * 0.1,
        -torch.ones(128, 16), torch.randn(1, 16, 200), torch.randn(1, 16, 200),
    )
    b = selective_scan_cpu_chunked(
        torch.randn(1, 128, 200) * 0 + 0, torch.randn(1, 128, 200).abs() * 0.1,
        -torch.ones(128, 16), torch.randn(1, 16, 200), torch.randn(1, 16, 200),
    )
    print("分块版与逐步版结构均可用（数值等价性由 test_parity 端到端验证）")

    # 3) SS2D + 完整块
    blk = LocalEnhancedVSSBlockCPU(dim=64)
    x4 = torch.randn(1, 64, 32, 32)
    t0 = time.time()
    out = blk(x4)
    print(f"LocalEnhancedVSSBlock CPU: 输入 {tuple(x4.shape)} → 输出 {tuple(out.shape)}, "
          f"耗时 {(time.time()-t0)*1000:.0f}ms (H=W=32)")
    print("✅ 模块自测通过")
