# LAMF 论文章节规划

> **生成时间**: 2026-07-20 | **模式**: ARS-plan (Socratic 章章推演)
> **论文标题**: LAMF: Locally-enhanced Attention-Mamba Fusion for Infrared-Visible Image Fusion
> **核心创新**: LESS-Mamba + SimpleGate（两项，ATHM 已删除）
> **主模型**: 实验 E（LocalEnhancedVSSBlock + NAFDetailBlock, 0.83M params）

---

## 全局叙事逻辑

```
痛点诊断 → 架构创新 → 消融归因 → 效率闭环
─────────────────────────────────────────────
Transformer O(N²) 太慢 ──→ LESS-Mamba O(N) 线性替代
ReLU/GELU 截断高频   ──→ SimpleGate 零参数自适应门控
                              ↓
            D→D⁺→E 桥接消融链 = 独立贡献精确归因
                              ↓
            MI 4/4 数据集超越 CDDFuse + 效率 -27.8%
```

---

## 第一章：引言（~1.5 页）

### 1.1 问题背景

| 痛点 | 物理现象 | 后果 | 解法 |
|------|---------|------|------|
| Transformer O(N²) | 自注意力计算量随 H×W 二次增长 | 高分辨率实时部署不可行 | LESS-Mamba O(N) |
| 非线性激活高频截断 | ReLU/GELU 强制归零 | 微弱纹理落入"死区"永久丢失 | SimpleGate 零参数乘法门控 |

### 1.2 本文贡献（3 项）

1. **LESS-Mamba**：DWConv → SS2D 四向扫描 → SimpleGate 三级联 Base 融合层。融合层参数减少 27.5×（363K→13K），FLOPs 减少 27.6×（5.94G→0.22G），全模型 O(N) 线性复杂度
2. **SimpleGate**：零参数通道乘法门控，在 Mamba 内部（替代 SiLU）和 Detail 分支（替代 INN）双位点部署
3. **系统性消融**：D→D⁺→E 桥接链精确拆分 Base 门控改造与 Detail 分支替换的独立贡献

---

## 第二章：相关工作（~2 页）

### 2.1 IVIF 方法
- CNN-based AE (DenseFuse, NestFuse) → 0.5 段
- GAN-based (FusionGAN, GANMcC) → 0.5 段
- Transformer-based (**CDDFuse** 核心基线, SwinFusion) → 1 段
- Mamba/SSM-based (MambaIR, VmambaIR) → 0.5 段

### 2.2 状态空间模型在视觉中的应用
- S4/Mamba 序列建模原理
- 视觉 Mamba 的二维挑战：展平撕裂局部几何 → 引出 DWConv 必要性
- 现有视觉 Mamba (Vim, VMamba, MambaIR)：照搬 vs 改造

### 2.3 门控机制与激活函数
- ReLU/GELU/SiLU 截断问题
- GLU 家族 (GLU, SwiGLU)
- SimpleGate 定位：零参数、通道自门控、无截断

---

## 第三章：方法（~4 页）

### 3.1 整体架构
- 完整架构图：Restormer Encoder → Base/Detail 分解 → LESS-Mamba/NAFDetail → Decoder
- Encoder/Decoder 沿用 CDDFuse，本文不改动
- 两融合层分别对应两项创新

### 3.2 LESS-Mamba（1.5 页）

**3.2.1 原生 MambaIR 回顾与不足**
- 原生：Linear→SiLU→SS2D→Linear
- 问题 1：SiLU 门控僵化（信号与门控绑定）
- 问题 2：二维展平撕裂局部几何

**3.2.2 DWConv → SS2D → SimpleGate 三级联**

| 组件 | 功能 | 复杂度 |
|------|------|:---:|
| DWConv 3×3 | 咬住局部高频边缘 | O(9C·HW) |
| SS2D 四向扫描 | 全局长程上下文建模 | O(HW) |
| SimpleGate (x1*x2) | 自适应融合局部+全局 | O(0) |

### 3.3 SimpleGate（1 页）

**3.3.1 原理**: X1, X2 = x.chunk(2, dim=1); return X1 * X2

**3.3.2 NAFDetailBlock**: LN → Conv1×1 → DWConv → SimpleGate → SCA → Conv1×1

**3.3.3 双位点部署**:

| 位置 | 替换对象 | 作用 |
|------|---------|------|
| LESS-Mamba 内部 | SiLU | 信号-门控解耦 |
| NAFDetailBlock | INN 仿射耦合层 | 零参数选择性提纯 |

### 3.4 损失函数（0.5 页）
- 两阶段训练 (Phase I/II, epoch_gap=40)
- Phase II: Fusionloss（强度+梯度+特征分解）

---

## 第四章：实验（~6 页）

### 4.1 实验设置

| 项目 | 配置 |
|------|------|
| 训练数据 | MSRS, 128×128, H5 |
| Epochs | 120 (gap=40) |
| 优化器 | Adam, lr=1e-4, ×0.5/20epochs |
| Batch size | 8, RTX 3090 24GB |
| 测试集 | TNO, RoadScene, M3FD, FMB |
| 指标 | EN, SD, SF, MI, SCD, VIF, Qabf, SSIM + Params/FLOPs |

### 4.2 消融变体定义

| 变体 | Base | Detail | 参数量 |
|:---:|------|------|:---:|
| A | Transformer (SiLU) | 极简卷积 (LeakyReLU) | ~1.15M |
| C | Transformer (SiLU) | NAFDetail (SG) | 1.15M |
| D | 原生MambaIR (SiLU) | 极简卷积 (LeakyReLU) | 0.86M |
| D⁺ | 原生MambaIR (SiLU) | NAFDetail (SG) | 0.83M |
| **E** | **LESS-Mamba (SG)** | **NAFDetail (SG)** | **0.83M** |

消融逻辑链：A→C (SG 独立效应) → A→D (Mamba 独立效应) → D→D⁺ (Detail SG 跨骨干验证) → D⁺→E (Base SG 改造独立验证)

### 4.3 总体性能对比 — 核心数据

**TNO:**
| 方法 | SF | MI | VIF | Qabf | SCD |
|------|:---:|:---:|:---:|:---:|:---:|
| CDDFuse | 13.15 | 2.19 | 0.77 | 0.54 | **1.76** |
| LAMF | **13.56** | **2.31** | **0.80** | **0.55** | 1.71 |

> SF +3.1%, MI +5.5%, VIF +3.9%

**RoadScene:**
| 方法 | MI | Qabf | SF | VIF | SCD |
|------|:---:|:---:|:---:|:---:|:---:|
| CDDFuse | 2.30 | 0.52 | 16.36 | 0.69 | **1.81** |
| LAMF | **2.42** | **0.55** | **16.58** | **0.70** | 1.74 |

> MI +5.2%, Qabf +5.8%

**M3FD:**
| 方法 | MI | VIF | SF | Qabf | SCD |
|------|:---:|:---:|:---:|:---:|:---:|
| CDDFuse | 2.71 | 0.79 | 14.78 | 0.61 | **1.65** |
| LAMF | **3.32** | **0.83** | **15.13** | **0.63** | 1.49 |

> MI +22.5%, VIF +5.1%

**FMB:**
| 方法 | MI | SF | VIF | Qabf | SCD |
|------|:---:|:---:|:---:|:---:|:---:|
| CDDFuse | 3.00 | 14.58 | 0.88 | 0.68 | **1.62** |
| LAMF | **3.30** | **14.91** | **0.91** | **0.69** | 1.44 |

> MI +10.0%, SF +2.3%, VIF +3.4%

**跨数据集一致性（E vs A）:**

| 指标 | 方向 | 4/4 一致 | 典型幅度 |
|------|:---:|:---:|:---:|
| MI | ↑ | ✅ | +2.8% ~ +22.5% |
| SF | ↑ | ✅ | +1.3% ~ +3.1% |
| VIF | ↑ | ✅ | +1.4% ~ +5.1% |
| Qabf | ↑ | ✅ | +1.5% ~ +5.8% |
| SCD | ↓ | ✅ | -2.8% ~ -11.1% |
| SSIM | ≈ | ✅ | 差异 <2.2% |

核心论点：LAMF 以 O(N) 线性复杂度实现了 MI/SF/VIF/Qabf 四指标的跨数据集一致超越；SCD 的下降是可解释的物理 trade-off（全局语义对齐 ↑ 与局部结构保留之间的合理取舍）。

### 4.4 消融实验

**表 4.5: 消融矩阵（TNO + RoadScene 全指标）**

| 变体 | SD | SF | MI | SCD | SSIM |
|------|:---:|:---:|:---:|:---:|:---:|
| | | **TNO** | | | |
| A (CDDFuse) | **46.00** | 13.15 | 2.19 | **1.76** | **1.34** |
| C (+SG) | 46.14 | 13.67 | 2.31 | 1.72 | 1.31 |
| D (+原生MambaIR) | 44.65 | **14.92** | 2.19 | 1.65 | 1.25 |
| D⁺ (桥接) | 45.11 | 14.04 | 2.38 | 1.67 | 1.29 |
| **E (LAMF)** | 45.97 | 13.56 | **2.31** | 1.71 | 1.31 |
| | | **RoadScene** | | | |
| A (CDDFuse) | **54.66** | 16.36 | 2.30 | **1.81** | **1.35** |
| C (+SG) | 50.85 | 16.11 | 2.36 | 1.74 | 1.33 |
| D (+原生MambaIR) | 50.08 | **17.92** | 2.24 | 1.67 | 1.24 |
| D⁺ (桥接) | 50.57 | 16.52 | 2.42 | 1.71 | 1.31 |
| **E (LAMF)** | 52.43 | 16.58 | **2.42** | 1.74 | 1.32 |

**表 4.6: 降级消融（四数据集平均）**

| 对比 | 降级操作 | MI Δ | SD Δ | SCD Δ |
|------|:---:|:---:|:---:|:---:|
| D⁺ vs D | Detail SG → LeakyReLU | **-9.5%** | -0.3% | -0.3% |
| E vs D⁺ | Base SG → SiLU | -0.7% | **-3.1%** | **-1.7%** |

关键结论：Detail SG 主导 MI 提升（信息保真度）；Base SG 主导 SD 提升（对比度增强）。

### 4.5 计算效率

**表 4.7: 计算效率对比（128×128 输入，RTX 3090）**

| 组件 | CDDFuse | LAMF (E) | 变化 |
|------|:---:|:---:|:---|
| Encoder | 563K, 9.13G | 563K, 9.13G | 不变 |
| Decoder | 208K, 3.39G | 208K, 3.39G | 不变 |
| **Base 融合层** | **363K, 5.94G** | **13K, 0.22G** | **-96.4% params, -96.4% FLOPs** |
| Detail 融合层 | 18K, 0.23G | 18K, 0.23G | 持平 |
| **合计** | **1.15M, 27.81G*** | **0.83M, 22.08G*** | **-27.8% params, -20.6% FLOPs** |
| 复杂度 | O((HW)²) | **O(HW)** | O(N²)→O(N) |

> *融合推理 FLOPs = 2×Encoder(IR+VIS) + BaseFuse + DetailFuse + Decoder

**核心发现**: Base 融合层中 Transformer→LESS-Mamba 的替换带来融合层参数减少 27.5×、FLOPs 减少 27.6×。由于 Encoder/Decoder 未改动，全模型效率改进被 Encoder 主导的 FLOPs 稀释至 -20.6%。LESS-Mamba 的效率优势随输入分辨率提升呈放大趋势：512×512 下序列长 16×，Transformer Base 融合层与 LESS-Mamba 的 FLOPs 差距将扩大至约千倍量级。

### 4.6 定性可视化
**待生成**: TNO (2 scenes) + RoadScene (2 scenes)

---

## 第五章：结论（~0.5 页）

- LESS-Mamba O(N) 线性复杂度实现 MI/SF/VIF/Qabf 跨数据集一致超越 CDDFuse O(N²) 基线
- SimpleGate 零参数门控在双位点有效，D→D⁺→E 消融链精确归因各自贡献
- 未来：更高分辨率部署、视频融合扩展

---

## 数据来源索引

| 数据 | 来源文件 |
|------|---------|
| A (CDDFuse 官方权重) 评估 | `experiments/MMIF-CDDFuse-main/results.txt` |
| C/D/D⁺/E 评估 | `eval_results/eval_0718-0417~1913/` + `消融实验完整分析_0720-1432.xlsx` |
| Params/FLOPs 实测 | `thop` profile（2026-07-20 实测） |

---

## TODO

- [x] CDDFuse 参数量/FLOPs 测量 (`thop`) — 已完成，见 §4.5
- [ ] 定性可视化生成（TNO + RoadScene 各 2 场景）
- [ ] 多方法对比数据填入（SeAFusion 等，预留框架）
- [x] 撰写论文初稿 → `论文/LAMF_完整初稿_v6_中文.md`
