# LAMF服务器端复现与效率评估任务交接文档
## ——统一指标、参数量、FLOPs、显存、延迟、FPS与分辨率扩展

> 目标：在服务器端对每一个 IVIF 方法进行统一、可追溯的推理与效率测试。
>
> 核心原则：**不修改作者模型核心结构，不伪造权重；统一评估条件，统一输出格式，统一效率测量方法。**

---

# 1. 当前已确定的实验方法

建议当前实验库登记：

```text
CDDFuse
ITFuse
STFNet（若已成功复现）
MAEFuse
SFDFuse
FusionMamba（补充参考，不作为Transformer核心论据）
LAMF
```

FefDM-Transformer：

```text
Not Reproduced
Reason:
No publicly available official project/repository satisfying
official-code + official-checkpoint hard requirement.
```

不要为了得到结果自行重构成“官方复现”。

---

# 2. 固定数据集

服务器路径：

```text
/home/hyhz/HT/test_img/<Dataset>/
```

| Dataset | Pairs | Input |
|---|---:|---|
| TNO | 25 | 原生分辨率 |
| RoadScene | 50 | JPG、尺寸不统一 |
| M3FD | 300 | 1024×768 |
| FMB | 280 | 800×600 |

总计：

```text
655 pairs
```

IR/VIS必须按去扩展名后的文件名动态匹配。

---

# 3. 固定指标

统一计算：

```text
EN
SD
SF
MI
SCD
VIF
Qabf
SSIM
```

统一实现：

```text
/home/hyhz/HT/code/MY-CDDFuse/utils/Evaluator.py
```

调用：

```text
Evaluator
```

参数顺序：

```text
(fused, ir, vi)
```

VIF必须使用：

```text
Evaluator.VIFF
```

不要调用模块级 `VIFF`。

---

# 4. 严格统一的图像处理管线

## 输入

```text
float32
[0,1]
```

## 需要padding时

```text
原始 H×W
↓
reflect pad
↓
模型前向
↓
裁回原始 H×W
```

禁止直接 resize。

## 输出

```text
model output
↓
min-max normalization
↓
×255
↓
uint8
↓
save PNG
↓
read back float32
↓
evaluate
```

禁止直接使用内存中的 uint8 做指标。

交接文档中已经验证：错误的 uint8 减法会显著破坏 SF，因此所有方法必须沿用这一流程。

---

# 5. 每个方法开工前的三项检查

## 5.1 查权重

```bash
find /home/hyhz/HT/对比方法/<Method>-main -maxdepth 2   \( -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" \)
```

## 5.2 导入冒烟

```bash
/home/hyhz/anaconda3/envs/my-cddfuse/bin/python -u -c "import <主模型模块>" 2>&1 | tail -8
```

## 5.3 上游仓库核对

有官方 GitHub时：

```bash
curl -s "https://api.github.com/repos/<作者>/<repo>/git/trees/main?recursive=1" | grep -o '"path": *"[^"]*\.py"'
```

如果报：

```text
ModuleNotFoundError
```

且缺失文件属于方法自身仓库：

```text
=> Official code incomplete
=> Stop reproduction
```

不要自行补齐后声称“官方复现”。

---

# 6. 每个方法最终必须生成的文件

```text
<Method>/
├─ outputs/
│  ├─ fused_images/
│  │  ├─ TNO/
│  │  ├─ RoadScene/
│  │  ├─ M3FD/
│  │  └─ FMB/
│  ├─ metrics/
│  │  ├─ per_image.csv
│  │  └─ average.csv
│  └─ runtime/
│     ├─ latency.csv
│     ├─ memory.csv
│     └─ throughput.csv
├─ checkpoint/
├─ source/
└─ experiment_meta.md
```

---

# 7. per_image.csv固定格式

必须：

```text
Image_ID
Dataset
EN
SD
SF
MI
SCD
VIF
Qabf
SSIM
```

不四舍五入。

另加可选字段：

```text
inference_time_ms
peak_memory_MB
```

但正式指标表只使用前8项评价指标。

---

# 8. average.csv

每个数据集：

```text
Dataset
N
EN
SD
SF
MI
SCD
VIF
Qabf
SSIM
```

另外输出全655张总体统计：

```text
ALL
655
...
```

建议同时保存：

```text
mean
std
median
min
max
```

其中论文主表主要报告 mean。

---

# 9. Params统计

每个模型必须记录：

```text
Total Params
Trainable Params
```

单位：

```text
K / M
```

建议同时区分：

```text
Encoder
Fusion
Decoder
```

对于LAMF尤其要记录：

```text
CDDFuse Base Fusion:
363K

LAMF Base Fusion:
13K
```

论文当前实验已经得到：

```text
CDDFuse:
1.15M total
27.81G FLOPs

LAMF:
0.83M total
22.08G FLOPs

Base:
363K -> 13K
5.94G -> 0.22G
```

这些数字必须重新通过统一脚本确认后再进入最终稿。

---

# 10. FLOPs统计原则

统一输入：

```text
[1, 1, H, W]
```

至少测试：

```text
128×128
256×256
512×512
```

对每个方法记录：

```text
FLOPs
MACs（如工具提供）
Params
```

注意：

不同FLOPs工具可能定义不同，因此：

1. 全部方法尽可能用同一工具；
2. 同一输入尺寸；
3. 同一batch；
4. 同一forward路径；
5. 如果模型存在预处理/后处理，论文中注明是否纳入。

---

# 11. Latency / FPS测量

统一：

```text
batch=1
single GPU
CUDA
```

推荐流程：

```text
warm-up 20~50次
↓
正式测量100~200次
↓
torch.cuda.synchronize()
↓
记录平均时间
```

公式：

```text
FPS = 1000 / latency_ms
```

同时记录：

```text
mean_latency_ms
std_latency_ms
peak_memory_MB
FPS
```

不要用：

```text
整个655张数据集运行时间
```

直接当成模型FPS。

数据集整体耗时受：

- IO
- Python
- PNG保存
- 指标计算

严重影响。

---

# 12. GPU显存

统一：

```python
torch.cuda.reset_peak_memory_stats()
```

前向完成后：

```python
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated()
```

记录：

```text
Peak Memory (MB)
```

同一环境、同一GPU、batch=1。

服务器当前统一GPU：

```text
NVIDIA RTX 3090 24GB
```

---

# 13. 真实端到端运行时间

另外保存：

```text
dataset_runtime.csv
```

字段：

```text
Method
Dataset
N
Total_Inference_Time
Average_Inference_Time
```

但是必须明确：

```text
Inference time only
```

或者：

```text
Inference + I/O
```

两者不能混写。

论文效率表优先使用：

```text
pure model inference
```

---

# 14. 分辨率扩展实验

LAMF的重要证据之一是：

```text
O(N)
vs
O(N²)
```

建议测试：

```text
128×128
256×256
512×512
```

记录：

```text
Resolution
N=H×W
FLOPs
Latency
Peak Memory
FPS
```

最终画：

```text
Resolution vs FLOPs
Resolution vs Latency
Resolution vs Memory
```

---

# 15. 理论复杂度与实际复杂度必须分开

论文中分别写：

### Theory

```text
Transformer Attention:
O(N²)

LESS-Mamba:
O(N)
```

### Measurement

实际报告：

```text
FLOPs
latency
memory
FPS
```

不要用：

```text
O(N)
```

直接推断“实际FPS一定更高”。

实际速度还受到：

- kernel实现；
- GPU并行效率；
- memory bandwidth；
- CUDA版本；
- 算子融合

影响。

---

# 16. Efficiency Pareto实验

建议生成：

## 图A：FLOPs vs MI

x：

```text
FLOPs ↓
```

y：

```text
MI ↑
```

## 图B：FLOPs vs SF

x：

```text
FLOPs ↓
```

y：

```text
SF ↑
```

## 图C：Latency vs MI

x：

```text
Latency ↓
```

y：

```text
MI ↑
```

LAMF如果位于Pareto前沿，应重点强调。

---

# 17. 四个数据集的效率不要重复测完整655张

效率实验可以使用：

```text
代表性统一输入
```

例如：

```text
128×128
256×256
512×512
```

这样避免IO和数据内容造成干扰。

质量实验：

```text
655 pairs
```

效率实验：

```text
controlled synthetic-size benchmark
```

两者逻辑分开。

---

# 18. 消融实验

LAMF至少：

```text
Baseline / CDDFuse
w/o DetailGate
w/o Mamba
w/o LocalDW
LAMF Full
```

并统一：

```text
TNO
RoadScene
M3FD
FMB
```

主要观察：

| Variant | 核心问题 |
|---|---|
| w/o DetailGate | SimpleGate对高频纹理的贡献 |
| w/o Mamba | Mamba骨干的贡献 |
| w/o LocalDW | 局部几何先验贡献 |
| Full | 综合效果 |

---

# 19. LAMF需要额外做的专项分析

## 19.1 Base融合层

记录：

```text
Parameter count
FLOPs
latency
```

重点：

```text
CDDFuse Transformer
vs
LESS-Mamba
```

---

## 19.2 Detail融合层

比较：

```text
INN
vs
NAFDetailBlock + SimpleGate
```

记录：

```text
Params
FLOPs
```

这样可以证明你的双位点改造不是单一Mamba替换。

---

# 20. 注意力 / SS2D机制可视化数据

如果需要绘制：

```text
Attention Map
SS2D response
```

不要凭图片软件画示意热图。

必须由模型内部forward hook导出：

```text
feature
attention
gate
state/response
```

保存：

```text
npz / pt / png
```

再进行统一可视化。

---

# 21. 实验元数据 experiment_meta.md

每个方法必须记录：

```text
Paper:
Year:
Official Repository:
Commit:
Checkpoint:
Checkpoint Source:
Environment:
Python:
PyTorch:
CUDA:
GPU:
Input Range:
Input Order:
Padding:
Normalization:
Output Postprocess:
Datasets:
Metrics:
Run Command:
Runtime:
Status:
```

---

# 22. 最终总实验表

服务器最终生成：

```text
master_results.csv
```

字段：

```text
Method
Year
Category
Dataset
N
EN
SD
SF
MI
SCD
VIF
Qabf
SSIM
Params_M
FLOPs_G
Latency_ms
FPS
PeakMemory_MB
Reproducibility_Status
Checkpoint
```

这张表是论文所有实验表格的数据唯一来源。

---

# 23. 当前项目状态

### 已完成基础设施

- 655对测试数据；
- 统一指标管线；
- 原生分辨率评估；
- padding/crop；
- 保存后读回；
- CSV；
- FusionMamba完整复现。

当前交接文档记录：FusionMamba 655张全部跑通，两次TNO结果逐字节一致。后续方法应沿用相同实验原则。

### 当前重点

```text
P0: FefDM标记不可复现，不再投入
P1: 所有已有方法结果统一归档
P2: 参数/FLOPs/Latency/Memory/FPS
P3: 128/256/512扩展
P4: 消融
P5: Master Results
P6: 论文表格自动生成
```

---

# 24. 最终原则

1. **统一评估，不统一模型实现。**
2. **官方权重优先，不自行重训练替代官方权重。**
3. **代码缺失就停止，不伪造复现。**
4. **质量指标与效率指标分开测。**
5. **理论复杂度和实际速度分开表述。**
6. **所有论文数字必须可以追溯到CSV。**
7. **任何最终论文表格都从master_results.csv生成。**


---

# 25. AI真实性、指标真实性与防幻觉强约束（最高优先级）

> 本节为服务器实验的最高优先级规则。任何AI生成内容、人工猜测数字或无法追溯的数据都不得进入论文正式结果。

## 25.1 AI绝对禁止生成实验数字

AI不得：

- 猜测指标；
- 根据同类论文“推算”缺失指标；
- 根据趋势补齐某个模型；
- 修改异常数字使表格更好看；
- 把论文README中的数字冒充当前统一管线复现结果；
- 把单张图片指标冒充数据集均值；
- 把官方论文数字与本地复现数字混在同一列。

**唯一合法来源：服务器实际产生的CSV/日志。**

正式论文数字必须满足：

```text
paper table
← master_results.csv
← method average.csv
← per_image.csv
← actual inference output
```

---

## 25.2 结果溯源链

每一个正式数字必须可追溯：

```text
论文Table
→ master_results.csv
→ Dataset
→ Image_ID集合
→ per_image.csv
→ fused PNG
→ checkpoint
→ source commit
→ run command
→ environment
```

若无法完成这条链，数字只能标记为：

```text
UNVERIFIED
```

不得进入正式论文表。

---

## 25.3 不允许“为了公平”手动修正数字

禁止：

```text
加减常数
四舍五入后再平均
手工校正
用论文均值替换本地均值
删除不利样本后重新平均
不同方法使用不同数据量
```

允许：

```text
统一读取
统一公式
统一dtype
统一padding
统一保存-读回
统一mean
```

---

## 25.4 每个模型必须保存实验元数据

固定：

```text
experiment_meta.json
```

建议：

```json
{
  "method": "",
  "paper": "",
  "year": "",
  "repository": "",
  "commit": "",
  "checkpoint": "",
  "checkpoint_sha256": "",
  "python": "",
  "pytorch": "",
  "cuda": "",
  "gpu": "",
  "dataset_root": "",
  "num_images": 655,
  "input_order": "",
  "input_range": "[0,1]",
  "padding": "reflect",
  "output_postprocess": "",
  "evaluator": "",
  "command": "",
  "timestamp": ""
}
```

其中 `checkpoint_sha256` 用于确认复现实验没有偷偷更换权重。

---

## 25.5 权重真实性

每个方法必须记录：

```text
官方仓库
checkpoint文件名
checkpoint来源
SHA256
```

禁止：

> “看起来像官方权重”的第三方权重。

如果只能找到第三方checkpoint：

```text
Status = THIRD_PARTY_CHECKPOINT
```

不得写：

```text
Official reproduced result
```

---

## 25.6 指标计算真实性

所有8项指标必须来自同一个统一Evaluator：

```text
EN
SD
SF
MI
SCD
VIF
Qabf
SSIM
```

必须保存：

```text
per_image.csv
average.csv
```

平均值必须由：

```text
per_image.csv
```

程序重新计算。

禁止直接复制README/论文表格数字。

---

## 25.7 数值异常检测

每次全量测试后自动运行 sanity check：

```text
N == 655
每个Dataset数量正确
无NaN
无Inf
无重复Image_ID
无缺失Image_ID
所有指标有限值
输出图数量 == 655
```

另外检查：

```text
min <= mean <= max
std >= 0
```

对于理论上有明确范围的指标，增加范围检查。

任何异常必须生成：

```text
anomaly_report.txt
```

先调查原因，再决定是否继续。

---

## 25.8 SSIM等指标的特别核验

不要仅根据“越大越好”假设结果合理。

如果出现：

```text
SSIM > 1
Qabf overflow
NaN
异常极端值
```

必须：

1. 定位具体 Image_ID；
2. 查看原始融合PNG；
3. 重算该指标；
4. 检查 Evaluator 输入范围；
5. 检查dtype；
6. 检查实现是否符合当前项目规定；
7. 记录解释。

**不能为了符合常识而强行clip。**

---

## 25.9 FLOPs真实性

FLOPs必须记录：

```text
tool
version
input_size
batch_size
forward_path
special_op_handling
```

不能：

> 直接把论文里的FLOPs写进当前实验结果。

如果某模型的特殊算子无法被工具正确统计：

```text
FLOPs = NOT VERIFIED
```

不要伪造估算值。

理论复杂度：

```text
O(N²)
O(N)
```

和实测FLOPs是两列不同证据，不得相互替代。

---

## 25.10 Latency真实性

正式延迟必须：

```text
GPU synchronize
warmup
multiple repetitions
mean
std
```

推荐记录：

```text
warmup = 30
repeats = 100
batch = 1
```

具体值可以根据模型调整，但必须在 `experiment_meta` 中固定记录。

不得：

> 用整套数据集耗时直接当FPS。

---

## 25.11 参数量真实性

Params必须从实际加载模型统计：

```text
sum(p.numel() for p in model.parameters())
```

必须确认：

```text
checkpoint loaded successfully
missing keys
unexpected keys
```

建议记录：

```text
missing_keys
unexpected_keys
```

正式实验必须：

```text
missing_keys == 0
unexpected_keys == 0
```

除非论文原模型明确允许，否则不能忽略异常key。

---

## 25.12 分辨率扩展真实性

128/256/512实验必须实际运行或明确标记：

```text
MEASURED
```

和：

```text
THEORETICAL
```

严格区分。

如果512×512只是根据O(N²)/O(N)推算：

```text
THEORETICAL
```

绝不能写成：

```text
measured FLOPs
```

如果实际运行：

```text
MEASURED
```

两种数据可以同时展示，但必须标注。

---

## 25.13 AI辅助实验分析限制

AI可以：

- 读取CSV；
- 计算均值/标准差/排名；
- 生成统计图；
- 查找异常；
- 自动生成实验报告草稿；
- 根据真实数据提出解释假设。

AI不得：

- 生成不存在的实验数据；
- 修改实验数据；
- 隐瞒不利结果；
- 删除异常样本以提高结果；
- 自动选择“看起来最漂亮”的数字作为论文结果；
- 把相关性描述成因果；
- 把视觉现象描述成机制证据而没有实验支持。

---

## 25.14 正式结果分级

建议每个结果标记：

```text
VERIFIED
REPRODUCED
THEORETICAL
UNVERIFIED
FAILED
```

论文主表只允许：

```text
VERIFIED / REPRODUCED
```

效率理论分析可以使用：

```text
THEORETICAL
```

但必须明确标注。

`UNVERIFIED / FAILED` 不得进入主结果表。

---

## 25.15 Master Results写入前的最终门禁

只有满足以下条件才能写入：

```text
[ ] 官方/明确来源checkpoint已记录
[ ] checkpoint SHA256已记录
[ ] source commit已记录
[ ] 代码导入正常
[ ] checkpoint加载无missing/unexpected keys
[ ] 655张完整跑通
[ ] Dataset数量正确
[ ] per_image.csv存在
[ ] average.csv由per_image.csv计算
[ ] 无NaN/Inf
[ ] 指标经过sanity check
[ ] 参数量真实统计
[ ] FLOPs来源明确
[ ] Latency使用GPU同步
[ ] Memory测量方法一致
[ ] experiment_meta.json完整
```

**任何一项不满足，状态保持为 UNVERIFIED，不得进入论文最终结果。**

---

# 26. AI给服务器端任务时必须附带的“防幻觉指令”

把下面文本作为所有后续AI任务的system/task前缀：

```text
STRICT REPRODUCIBILITY AND DATA-INTEGRITY MODE

You are processing real scientific experiments.

ABSOLUTE RULES:
1. Never invent, estimate, interpolate, or hallucinate experimental numbers.
2. Never modify an experimental result to make it more favorable.
3. Never copy a paper/README metric and label it as our reproduced result.
4. Use only the supplied CSV, logs, checkpoints, and actual inference outputs.
5. Every number must be traceable to a file and command.
6. If information is missing, output "UNVERIFIED" and stop; do not guess.
7. Never silently repair NaN, Inf, overflow, missing images, or abnormal metrics.
8. Never delete inconvenient samples unless the exclusion rule was predefined and documented.
9. Distinguish MEASURED from THEORETICAL values at all times.
10. Preserve raw experimental data; analysis must be read-only.
11. If checkpoint loading has missing/unexpected keys, do not claim successful reproduction.
12. When uncertain, ask for the original file/log/source instead of filling the gap.

Scientific priority:
truth > completeness > presentation > convenience.
```

---

# 27. 推荐自动化审计文件

服务器端最终增加：

```text
audit/
├─ reproducibility_audit.json
├─ anomaly_report.txt
├─ checkpoint_hashes.csv
├─ environment_report.txt
├─ metric_sanity_report.csv
└─ run_manifest.csv
```

其中：

`run_manifest.csv`

建议：

```text
Method
Dataset
N
Command
Checkpoint
Checkpoint_SHA256
StartTime
EndTime
Status
```

这样任何审稿实验都可以反向追踪。
