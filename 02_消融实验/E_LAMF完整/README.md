# LAMF 主模型训练（干净版，无 Mask）

## 架构

```
Base 融合:   LocalEnhancedVSSBlock  (LESS-Mamba, SimpleGate)
Detail 融合:  DetailFeatureExtraction (NAFDetailBlock, SimpleGate)
损失函数:     Fusionloss only (intensity + gradient + decomp)
```

## 一键运行

```bash
cd /home/hyhz/HT/code/MY-CDDFuse/experiments/LAMF_Full

# 训练 + 评估（全流程）
python run_all.py

# 仅训练
python run_all.py --train-only

# 仅评估（自动找最新权重）
python run_all.py --eval-only

# 指定权重评估
python run_all.py --ckpt checkpoints/LAMF_Full_Final_xxx.pth

# 保留融合图像不删除
python run_all.py --keep-fused
```

## 输出

| 产物 | 路径 | 说明 |
|------|------|------|
| 训练权重 | `checkpoints/` | Epoch 40 快照 + Final 权重 |
| 评估 Excel | `results/eval_xxx.xlsx` | 原始评估数据 |
| 实验报告 | `results/实验报告_xxx.xlsx` | 含指标 + 配置 + 消融框架 |

## 手动分步运行

```bash
# 步骤 1: 训练
python train.py

# 步骤 2: 评估
python ../../eval_all.py --ckpt checkpoints/LAMF_Full_Final_xxx.pth --name LAMF_Full_Clean --variant E
```

## 与 train_my.py 的区别

| 项目 | `train_my.py` (项目根) | `train.py` (本实验) |
|------|------------------------|---------------------|
| Mask | ✓ 有 (GradientConflictMasker) | ✗ 无 |
| 损失 | MaskAwareFusionLoss + MI Loss | Fusionloss only |
| 权重保存 | `/home/hyhz/HT/cddfuse_data/models/` | `checkpoints/` |
| 用途 | 旧版含 Mask 实验 | 论文两创新洁净版本 |
