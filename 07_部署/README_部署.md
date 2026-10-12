# LAMF 香橙派 5 Plus 部署指南

> 面向零 RKNN 经验用户 | 更新: 2026-09-22

---

## 一、方案说明（先读这段）

### 为什么不用 NPU（RKNN）跑整个模型？

LAMF 的 Base 融合层使用 **Mamba（选择性状态空间扫描）**，其计算是**数据依赖的动态循环**：

- RKNN-Toolkit2 的 ONNX 转换**不支持**这种结构（无法静态化为计算图）
- 强行展开 16384+ 步的静态图 → 编译不可行
- RK3588 上跑 SSM 的已知方案需要**手写分块 scan 的 C 内核**（数周工作量），且实测在 130M 以下的小模型上 NPU 反而比 CPU 慢（0.7×）。LAMF 的融合层仅 45K 参数，投入产出比极低

### 采用的方案：纯 CPU 推理

| 组件 | 运行位置 | 说明 |
|------|:---:|------|
| Restormer Encoder/Decoder | CPU (PyTorch) | 纯 Conv/Attention，无需 CUDA |
| LESS-Mamba 融合层 | CPU (纯 PyTorch 实现) | 替代了 CUDA 专用算子 |
| NAFDetailBlock | CPU (PyTorch) | 纯 Conv |

**关键优势**：**权重完全复用**，无需重新训练、无需转换。`mamba_cpu.py` 的参数结构与官方 `mamba_ssm.Mamba` 完全一致（已用 `strict=True` 验证零缺失零多余）。

---

## 二、本机（服务器）快速验证

```bash
cd /home/hyhz/HT/code/MY-CDDFuse/deploy

# 1) 自测：模型构建 + 权重加载 + 前向
python lamf_cpu.py

# 2) 数值正确性校验（selective scan 与官方参考实现 bit-exact）
python test_parity.py --strict

# 3) 单张推理（--size 128 限制长边以控制耗时）
python infer.py --ir /path/ir.png --vis /path/vis.png --out fused.png --size 128

# 4) 批量推理
python infer.py --ir-dir data/ir --vis-dir data/vi --out-dir results --size 128
```

---

## 三、迁移到香橙派 5 Plus

### 3.1 传输文件

把整个 `deploy/` 目录传到香橙派（约 3.5MB，主要是权重）：

```bash
# 在服务器执行（替换 <用户名>@<香橙派IP>）
scp -r deploy/ <用户名>@<香橙派IP>:~/lamf_deploy/
```

或打包后传输：
```bash
tar czf lamf_deploy.tar.gz deploy/
# 传输 lamf_deploy.tar.gz 后在香橙派上: tar xzf lamf_deploy.tar.gz
```

### 3.2 香橙派环境要求

| 依赖 | 版本要求 | 安装方式 |
|------|---------|---------|
| Python | ≥ 3.8 | 系统自带 |
| PyTorch (CPU) | ≥ 1.13 | 见下 |
| numpy | 任意 | `pip install numpy` |
| opencv-python | 任意 | `pip install opencv-python` |

**PyTorch ARM64 安装**（若无）：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 或使用香橙派官方提供的 wheel
```

> `einops`/`timm` 仅在导入 `net.py` 时需要（Encoder/Decoder 用到），如缺失：
> `pip install einops timm`

### 3.3 运行

```bash
cd ~/lamf_deploy/deploy
python infer.py --ir ir.png --vis vis.png --out fused.png --size 128
```

---

## 四、性能预期（重要）

### 快速参考

| 输入分辨率 | 服务器(x86) | 香橙派预估 | 推荐场景 |
|:---:|:---:|:---:|------|
| 128×128 | ~3.5 s | 10~20 s | **✅ 推荐** |
| 256×256 | ~12 s | 40~70 s | 精度优先 |
| 原生分辨率 | 100~280 s | 不可用 | ❌ 仅离线批处理 |

> 详细的三档分辨率 x 四数据集基准数据见 `benchmark/效率分析报告.md`

### 基准测试工具

```bash
# 完整测试（选图 + 三档分辨率 + 效率报告）
python benchmark.py

# 快速测试（仅 128/256）
python benchmark.py --sizes 128 256

# 自定义每数据集图像数
python benchmark.py --per-dataset 4
```

产出：
```
benchmark/
├── demo_inputs/{ir,vi}/          # 选取的 demo 输入图像
├── fused/{128,256,native}/<数据集>/  # 各分辨率融合结果
├── benchmark_results.csv         # 逐图逐分辨率耗时
└── 效率分析报告.md                # 汇总分析（含香橙派预估）
```

### 耗时构成（128×128）

| 组件 | 耗时 | 占比 |
|------|:---:|:---:|
| Encoder (×2, IR+VIS) | 2.7 s | 35% |
| **LESS-Mamba 扫描** | **4.7 s** | **61%** |
| Decoder | 0.3 s | 4% |

**瓶颈是 Mamba 扫描**：序列长度 = H×W（128×128 时 16384 步 × 4 个扫描方向）。

### 香橙派上的预期

RK3588（4×A76 + 4×A55）单核性能约为服务器 x86 的 40-60%，估算：

| 分辨率 | 预计耗时 |
|:---:|:---:|
| 128×128 | **20 ~ 40 s/张** |
| 256×256 | 60 ~ 120 s/张 |

> 如果这个速度不满足需求，可选优化路径见第六节。

---

## 五、数值一致性（已验证）

### 正确性基准

| 对比 | 最大差异 | 结论 |
|------|:---:|:---:|
| CPU scan vs 官方 PyTorch 参考实现 | **0.0（bit-exact）** | ✅ 实现完全正确 |
| CPU vs GPU (CUDA 算子) 中间特征 | ~16% | CUDA 算子浮点行为差异（长序列放大） |
| **CPU vs GPU 融合图像指标** | MI/SF/VIF/Qabf 偏差 **<6%**（VIF/Qabf <2.2%） | ✅ **融合质量一致** |

**结论**：CPU 版与原 GPU 版的融合图像质量基本一致，可放心部署。

---

## 六、可选优化路径

若需要更快的速度：

1. **降低输入分辨率**（最直接）
   - 128×128 → 8s；64×64 → 约 2-3s（香橙派上约 5-10s）

2. **启用并行扫描**（小幅收益）
   ```python
   from lamf_cpu import LAMF_CPU
   model = LAMF_CPU(use_chunked=True, chunk_size=512)   # Hillis-Steele 并行扫描
   ```
   实测加速约 1.5×，但有轻微数值差异（~1e-3）

3. **Encoder 上 NPU**（未来工作）
   - Encoder 占 35% 耗时且是标准 Conv/Attention，理论上可转 RKNN
   - 需要 `rknn-toolkit2==2.3.0`（与开发板 `librknnrt.so 2.3.0` 匹配）
   - 预期可省下 2.7s（128×128）

4. **模型级优化**（如需实时）
   - 知识蒸馏到无 Mamba 的轻量模型
   - 或对扫描方向做剪枝（如只用 2 个方向）

---

## 七、文件说明

| 文件 | 用途 |
|------|------|
| `mamba_cpu.py` | 纯 PyTorch 的 Mamba/SS2D 实现（替代 CUDA 算子） |
| `lamf_cpu.py` | LAMF 完整网络（CPU 版），含权重加载 |
| `infer.py` | 推理脚本（单张/批量） |
| `test_parity.py` | 数值对齐验证工具 |
| `weights/lamf_final.pth` | 模型权重（0.83M 参数，3.3MB） |

---

## 八、常见问题

**Q: 报错 `ModuleNotFoundError: No module named 'timm'`**
→ `pip install timm`（`net.py` 的 Encoder/Decoder 用到）

**Q: 报错 `ModuleNotFoundError: No module named 'mamba_ssm'`**
→ 不应该出现。本部署包**不需要** mamba_ssm。如果出现，检查是否误入了项目根目录的旧代码路径（`utils/lamf_mamba_layer.py` 才依赖它）。

**Q: 内存不足（MemoryError）**
→ 降低输入分辨率（`--size 128`），或减小 `chunk_size`。

**Q: 输出图像全黑/全白**
→ 检查输入是否为灰度单通道图像；融合输出已做 min-max 归一化。

**Q: 想验证 CPU 版输出是否正确**
→ 在服务器上跑 `python test_parity.py`，会对比 GPU 版与 CPU 版的指标差异。
