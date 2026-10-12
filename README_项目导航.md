# LAMF 项目导航

> 整理日期: 2026-10-12 | 用于快速定位「论文的每个数据/图 ↔ 对应的文件」

---

## 一、目录总览

```
MY-CDDFuse/
├── net.py                   ★ 网络定义（Encoder/Decoder/NAFDetailBlock/SimpleGate）
├── utils/                   ★ 工具库（lamf_mamba_layer.py = LESS-Mamba 实现）
├── eval_all.py              ★ 统一评估管线（GPU 批量推理+指标计算）
│
├── 01_模型代码/              训练与推理脚本
├── 02_消融实验/              5 个变体（A/C/D/D⁺/E），各含 训练脚本+权重+评估报告
├── 03_效率实验/              效率数据（服务器实测 + 香橙派部署基准）
├── 04_复现对比/              LAMF vs CDDFuse 完整复现结果（含 655 张融合图）
├── 05_论文/                  论文稿 + 写作素材 + 数据表
├── 06_图表/                  架构图 + 候选图 + 可视化工具
├── 07_部署/                  香橙派 5 Plus 部署包
├── 08_文档/                  交接文档 + 复现指南
└── 09_参考资料/              参考论文 PDF
```

---

## 二、★ 一对一对应表（论文 ↔ 文件）

### 论文正文

| 论文位置 | 内容 | 对应文件 |
|------|------|------|
| **摘要/引言** | 核心指标（MI+9.5%、SD+3.1%、Params -27.8%） | `04_复现对比/LAMF/outputs/metrics/average.csv`<br>`03_效率实验/master_results.csv` |
| **§3 方法** | 网络结构 | 根目录 `net.py`（整体）<br>`utils/lamf_mamba_layer.py`（LESS-Mamba） |
| **§4.1** 实验设置 | 数据集/指标/消融变体定义 | `02_消融实验/*/README.md`（若有）<br>`03_效率实验/规范输出_padded/experiment_meta.md` |
| **§4.2** 总体性能对比 | LAMF vs CDDFuse 四数据集指标 | `04_复现对比/LAMF/outputs/metrics/average.csv`<br>`04_复现对比/CDDFuse/outputs/metrics/average.csv` |
| **§4.3** 消融实验 | A/C/D/D⁺/E 消融矩阵 | `02_消融实验/{A,C,D,D+_桥接消融,E}*/results/*.xlsx`<br>`05_论文/数据/消融实验完整分析_0720-1432.xlsx` |
| **§4.4** 计算效率 | Params/FLOPs/Latency/Memory | `03_效率实验/规范输出_padded/runtime/{params,flops,latency,memory}.csv`<br>`03_效率实验/规范输出_padded/runtime/resolution_scaling.csv` |
| **§4.5** 下游任务 | （待补充检测实验） | — |

### 图表

| 论文图 | 内容 | 文件 |
|------|------|------|
| **图 1** | 整体架构总图 | `06_图表/主图/model_architecture_paper_4x1.pdf` |
| **图 2** | LESS-Mamba 模块细节 | `06_图表/模块图/part_b_architecture.pdf` |
| **图 3** | NAFDetailBlock 模块细节 | `06_图表/模块图/part_c_architecture.pdf` |
| **定性对比图** | 候选融合图（待排版） | `06_图表/候选图选图/`（含四联对比图 + 选图建议） |

### 关键数字溯源

| 数字 | 出处 |
|------|------|
| LAMF: 0.834M params / 22.08G FLOPs | `03_效率实验/规范输出_padded/params.csv`、`runtime/flops.csv` |
| CDDFuse: 1.188M params / 29.21G FLOPs | `04_复现对比/CDDFuse/outputs/runtime/{params,flops}.csv` |
| Base 融合层: 363K→45.8K params | 同上（`params.csv` 中 BaseFuse 行） |
| CDDFuse 官方复现基线（逐位一致） | `04_复现对比/CDDFuse官方代码/results.txt` |

---

## 三、各类别详解

### 01_模型代码/ — 训练与推理

| 文件 | 说明 |
|------|------|
| `train_lamf.py` | ★ 主模型（实验 E）训练脚本 |
| `train_baseline.py` | 基线（CDDFuse）训练脚本 |
| `run_all.py` | 一键训练+评估+出报告 |
| `test_IVF.py` | 单模型测试（CDDFuse 官方权重） |
| `test_env.py` | 环境自检 |
| `generate_paper.py` | 论文表格生成（旧版） |

> 训练数据: `/home/hyhz/HT/cddfuse_data/data/MSRS_train_imgsize_128_stride_200.h5`

### 02_消融实验/ — 5 个变体

| 目录 | 变体 | Base 融合层 | Detail 融合层 | 权重 |
|------|------|------|------|:---:|
| `A_CDDFuse基线` | A | Transformer(SiLU) | 极简卷积 | 1个 |
| `C_加SimpleGate` | C | Transformer(SiLU) | NAFDetail(SG) | 2个 |
| `D_加原生Mamba` | D | 原生MambaIR(SiLU) | 极简卷积 | 2个 |
| `D+_桥接消融` | D⁺ | 原生MambaIR(SiLU) | NAFDetail(SG) | 2个 |
| `E_LAMF完整` | **E（主模型）** | **LESS-Mamba(SG)** | **NAFDetail(SG)** | 2个 |

每个目录含：`train.py`（训练）、`run_all.py`（训练+评估）、`checkpoints/`（权重）、`results/`（评估报告 xlsx）

### 03_效率实验/

| 子目录 | 内容 | 来源 |
|------|------|------|
| `规范输出_padded/` | LAMF 规范输出（reflect pad 到 32）+ 完整报告（anomaly/sanity/stats） | 按交接文档规范 |
| `部署性能_香橙派/` | CPU 推理基准（128/256/原生 × 4 数据集）+ 效率分析报告 | `07_部署/benchmark.py` |
| `复现脚本/` | `run_full_eval.py`、`run_efficiency.py`、`build_master.py` | 可重跑 |
| `运行日志/` | 各次评估的日志 | — |
| `master_results.csv` | ★ 汇总表（Method/Dataset/8指标/Params/FLOPs/Latency/FPS） | — |
| `效率测量说明.md` | 测量协议 + 口径差异说明（**必读**） | — |

### 04_复现对比/

| 子目录 | 内容 |
|------|------|
| `LAMF/` | 655 张融合图 + 逐图指标 + 效率 CSV + 元数据（**无 padding 版本**） |
| `CDDFuse/` | 同上（CDDFuse 官方权重，8 指标与官方 results.txt 逐位一致） |
| `CDDFuse官方代码/` | MMIF-CDDFuse 官方仓库（含权重 CDDFuse_IVF.pth） |

> ⚠️ **重要差异**：`04_复现对比/` 与 `03_效率实验/规范输出_padded/` 的融合图**不完全相同**——前者按原生分辨率推理，后者做了 reflect padding（补齐到 32 的倍数）。**TNO/M3FD（尺寸可整除 32）结果一致，RoadScene/FMB 有差异**。论文引用时请统一口径。

### 05_论文/

| 子目录 | 内容 |
|------|------|
| `初稿/` | `LAMF_初稿_v1.md`、`.docx` |
| `写作素材/` | `写作风格.md`、`备忘录_摘要定稿.md`、`LAMF_论文章节规划.md` |
| `数据/` | `实验_E.xlsx`、`消融实验完整分析.xlsx`、`eval_full_LAMF.xlsx`、`指标.xlsx` |

### 06_图表/

| 子目录 | 内容 |
|------|------|
| `主图/` | 架构总图 PDF |
| `模块图/` | LESS-Mamba、NAFDetailBlock 细节图 |
| `候选图选图/` | 四联对比图（IR\|VIS\|CDDFuse\|LAMF）+ 选图建议 md |
| `viz_pipeline/` | 可视化工具（选图分析、融合图批量生成、指标对比） |
| `历史版本/` | 早期架构图 PNG |

### 07_部署/ — 香橙派 5 Plus

| 文件 | 说明 |
|------|------|
| `README_部署.md` | ★ 完整部署指南（零 RKNN 经验向） |
| `lamf_cpu.py` / `mamba_cpu.py` | 纯 PyTorch CPU 版网络（替代 CUDA 算子） |
| `infer.py` | 推理脚本（单张/批量） |
| `benchmark.py` | 性能基准测试 |
| `test_parity.py` | 数值对齐验证工具 |
| `weights/lamf_final.pth` | 主模型权重 |

---

## 四、快速上手

```bash
# 训练主模型（实验 E）
python 02_消融实验/E_LAMF完整/train.py

# 批量评估（GPU）
python eval_all.py

# 复现对比评估（LAMF 或 CDDFuse）
python "03_效率实验/复现脚本/run_full_eval.py" --method lamf

# 效率基准（GPU）
python "03_效率实验/复现脚本/run_efficiency.py" --method lamf

# 汇总 master_results.csv
python "03_效率实验/复现脚本/build_master.py"

# 香橙派部署推理（CPU）
python "07_部署/infer.py" --ir ir.png --vis vis.png --out fused.png --size 128
```

---

## 五、已废弃/归档

| 内容 | 位置 | 说明 |
|------|------|------|
| 早期架构图 | `06_图表/历史版本/` | 已被 `主图/` 的 PDF 取代 |
| CBRI 跨分支交互实验 | 已删除 | 三次尝试均失败，未纳入论文 |
| `repro_results.7z` / `fused_LAMF.7z` | 已删除 | 与已解压目录重复（513M） |
