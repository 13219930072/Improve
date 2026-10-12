# 实验索引

> 每个文件夹 = 一个独立消融实验。统一架构：`train.py` + `run_all.py`(一键训练+评估+报告) + `checkpoints/` + `results/`

---

## 消融矩阵

```
A  (基线):   Transformer         + SimpleDetailFusion       ← CDDFuse 原版
C  (+SG):    Transformer         + NAFDetailBlock(SG)       ← Detail 引入 SimpleGate
D  (+Mamba): OriginalVSSBlock    + SimpleDetailFusion       ← Base 换 MambaIR
D⁺ (桥接):  OriginalVSSBlock    + NAFDetailBlock(SG)       ← D→E 桥梁
E  (LAMF):   LocalEnhancedVSSBlock + NAFDetailBlock(SG)     ← 最终模型
```

## 实验论证链

```
创新1 (SimpleGate):      C vs A  → Detail SG 比 LeakyReLU 提升多少
创新2 (LESS-Mamba):      E vs D+ → LESS-Mamba SG 比 SiLU Mamba 提升多少
Mamba 替代 Transformer:  D vs A  → MambaIR 比 Transformer 提升多少
桥接验证:                D+ vs D → 在 Mamba 基线上，SG Detail vs 卷积 Detail
```

---

## 实验清单

| 实验 | 文件夹 | Base融合 | Detail融合 | 权重 | 一键脚本 | 状态 |
|------|--------|----------|------------|------|----------|:---:|
| **A** | `A/` | Transformer | SimpleDetailFusion | ✅ 已有 | `run_all.py` | ⬜ 待跑 |
| **C** | `C/` | Transformer | NAFDetail SG | ❌ 需训 | `run_all.py` | ⬜ 待跑 |
| **D** | `D/` | MambaIR SiLU | SimpleDetailFusion | ❌ 需训 | `run_all.py` | ⬜ 待跑 |
| **D+** | `D_plus/` | MambaIR SiLU | NAFDetail SG | ❌ 需训 | `run_all.py` | ⬜ 待跑 |
| **E** | `LAMF_Full/` | LESS-Mamba SG | NAFDetail SG | ✅ 已训 | `run_all.py` | ✅ 完成 |

### 附加实验

| 实验 | 文件夹 | 用途 | 状态 |
|------|--------|------|:---:|
| M3FD 重跑 | `M3FD_rerun/` | 实验 D (batch_size=4, NaN 排查) | ⬜ 待跑 |
| 损失消融 | `loss_ablation/` | Fusionloss / +MI / +Mask 对比 | ⬜ 待跑 |
| FMB 专项 | `FMB_eval/` | FMB 数据集单独评估 | ⬜ 待跑 |
| 效率分析 | `FLOPs/` | 全 5 变体 FLOPs + 参数量 | ⬜ 待跑 |

---

## 运行方式

### 一键批量运行（全部实验）

```bash
cd experiments
python run_all_experiments.py                    # 按推荐顺序跑全部 (A→C→D→D+)
python run_all_experiments.py --eval-only        # 仅评估 (需要权重已存在)
python run_all_experiments.py --only C,D_plus    # 仅跑指定实验
python run_all_experiments.py --skip D           # 跳过某个实验
python run_all_experiments.py --stop-on-error    # 遇错即停
```

### 单个实验运行

```bash
cd experiments/<实验名>
python run_all.py                 # 训练 + 评估 + 报告（全部自动）
python run_all.py --train-only    # 仅训练
python run_all.py --eval-only     # 仅评估（自动找最新权重）
```

### 实验 A 特例（已有权重）

```bash
cd experiments/A
python run_all.py                        # 直接用 CDDFuse_localtrain.pth 评估
python run_all.py --train                # 如果想从零重新训练
```

### 手动分步

```bash
cd experiments/<实验名>
python train.py                                                       # 步骤1: 训练
python ../../eval_all.py --ckpt checkpoints/xxx.pth --variant <X>    # 步骤2: 评估
```

---

## 公共路径

| 资源 | 路径 |
|------|------|
| 训练数据 | `/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5` |
| 测试图像 | `/home/hyhz/HT/test_img/` (TNO / RoadScene / M3FD / FMB) |
| A 权重(已有) | `/home/hyhz/HT/cddfuse_data/models/CDDFuse_localtrain.pth` |

---

## 运行顺序建议

```
1. FLOPs/measure_all.py            # 几分钟，确认所有变体架构正确
2. A/run_all.py                    # 几分钟 (eval only)，获得基线
3. D/run_all.py                    # ~几小时 (训练+评估)
4. D_plus/run_all.py               # ~几小时 (训练+评估) —— 核心消融
5. C/run_all.py                    # ~几小时 (训练+评估)
```

---

## 已有权重（跨实验参考）

| 实验 | 路径 | 备注 |
|------|------|------|
| A | `/home/hyhz/HT/cddfuse_data/models/CDDFuse_IVF.pth` | CDDFuse 官方 |
| A | `/home/hyhz/HT/cddfuse_data/models/CDDFuse_localtrain.pth` | 本地复现 |
| E | `LAMF_Full/checkpoints/LAMF_Full_Final_07-17-13-19.pth` | 干净版 LAMF |
| E(旧) | `/home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final_07-16-19-09.pth` | 含 Mask 旧版 |
| E(旧) | `/home/hyhz/HT/cddfuse_data/models/LAMF_Full_Final.pth` | 更早版本 |
