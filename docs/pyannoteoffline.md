# pyannote.audio 离线部署完整解决方案

**pyannote.audio speaker diarization 可以在完全离线环境中运行**，关键是正确下载模型文件并配置本地化配置文件。最新的 3.x 版本已经移除了 ONNX 依赖，采用纯 PyTorch 实现，大大简化了离线部署。核心解决方案是使用特定的文件命名规范和本地 YAML 配置文件来绕过 Hugging Face Hub 的在线验证。

官方维护者明确表示："整个认证过程不会阻止您在离线环境中使用官方 pyannote.audio 模型"，并在 GitHub 仓库中提供了专门的离线使用教程。

## 核心解决方案：本地配置文件方法

### 必需的模型文件下载

首先在有网络的环境中下载以下模型文件（需要 HuggingFace 账户和 token）：

1. **分割模型**：从 `pyannote/segmentation-3.0` 下载 `pytorch_model.bin` (5.7MB)
2. **嵌入模型**：从 `pyannote/wespeaker-voxceleb-resnet34-LM` 下载 `pytorch_model.bin` (26MB)

**关键文件命名规范**：
```bash
# 重命名下载的文件（命名至关重要！）
mv segmentation_pytorch_model.bin models/pyannote_model_segmentation-3.0.bin
mv wespeaker_pytorch_model.bin models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
```

文件名必须包含 `pyannote_model_` 前缀，这触发正确的加载机制。

### 创建离线配置文件

创建 `config.yaml` 文件：

```yaml
version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: models/pyannote_model_segmentation-3.0.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
```

### 目录结构要求

```
project/
├── models/
│   ├── pyannote_model_segmentation-3.0.bin
│   ├── pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
│   └── config.yaml
└── your_script.py
```

## Hugging Face Hub 在线验证绕过方法

### 环境变量配置

设置以下环境变量以确保完全离线运行：

```bash
# 禁用 HuggingFace Hub 联网
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 禁用其他在线功能
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 设置缓存目录（可选）
export HF_HOME="/path/to/offline/cache"
export HF_HUB_CACHE="/path/to/offline/cache/hub"
```

### Python 代码实现

```python
import os
from pathlib import Path
from pyannote.audio import Pipeline

# 设置离线环境
os.environ.update({
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'
})

def load_offline_pipeline(config_path):
    """加载离线 pipeline 的推荐方法"""
    config_path = Path(config_path)
    cwd = Path.cwd().resolve()
    
    # 切换到配置文件目录以处理相对路径
    os.chdir(config_path.parent)
    
    try:
        pipeline = Pipeline.from_pretrained(config_path)
        return pipeline
    finally:
        os.chdir(cwd)  # 切换回原目录

# 使用示例
pipeline = load_offline_pipeline("models/config.yaml")
diarization = pipeline("audio.wav")

# 输出结果
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
```

## 缓存设定和符号连结配置

### HuggingFace 缓存机制配置

如果您已经在在线环境中下载过模型，可以配置缓存目录：

```bash
# 设置 HuggingFace 缓存目录
export HF_HOME="/offline/cache/huggingface"
export HF_HUB_CACHE="/offline/cache/huggingface/hub"
export TRANSFORMERS_CACHE="/offline/cache/huggingface/hub"

# 创建缓存目录结构
mkdir -p "$HF_HOME"/{hub,datasets,assets}
```

### 符号连结设置

对于已存在的缓存，可以使用符号连结：

```bash
# 移动现有缓存到新位置
mv ~/.cache/huggingface /offline/cache/huggingface

# 创建符号连结
ln -s /offline/cache/huggingface ~/.cache/huggingface

# 或者直接设置环境变量指向新位置
export HF_HOME="/offline/cache/huggingface"
```

### 预下载脚本（在线环境中运行）

```python
from huggingface_hub import snapshot_download
import os

# 设置缓存目录
os.environ['HF_HOME'] = '/offline/cache/huggingface'

# 下载所需模型
models = [
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM"
]

for model in models:
    print(f"正在下载 {model}...")
    snapshot_download(
        repo_id=model,
        repo_type="model",
        use_auth_token="your_hf_token_here"
    )
```

## 完整离线部署步骤

### 步骤 1：在线准备阶段

1. **创建 HuggingFace 账户**并接受模型使用条件：
   - 访问 `pyannote/segmentation-3.0` 并接受条件
   - 访问 `pyannote/speaker-diarization-3.1` 并接受条件

2. **安装依赖**：
   ```bash
   pip install pyannote.audio>=3.1
   ```

3. **下载模型文件**：
   ```bash
   # 创建模型目录
   mkdir -p models/
   
   # 下载并重命名模型文件
   wget https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin \
        -O models/pyannote_model_segmentation-3.0.bin
   
   wget https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin \
        -O models/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
   ```

### 步骤 2：离线环境部署

1. **传输文件**到离线环境：
   - 模型文件
   - 配置文件
   - Python 脚本

2. **设置环境变量**：
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

3. **验证离线运行**：
   ```python
   from pyannote.audio import Pipeline
   
   # 测试加载
   pipeline = Pipeline.from_pretrained("models/config.yaml")
   print("离线加载成功！")
   ```

## 常见问题解决方案

### 问题 1：模型加载失败
**错误**：`KeyError` 或 `Model not found`
**解决方案**：
- 确认文件命名严格遵循 `pyannote_model_` 前缀
- 检查配置文件中的路径是否正确
- 使用绝对路径而不是相对路径

### 问题 2：ONNX 或 Protobuf 解析错误
**错误**：`INVALID_PROTOBUF` 或 ONNX runtime 错误
**解决方案**：
- 确保使用 pyannote.audio 3.1+ 版本
- 重新下载模型文件（可能文件损坏）
- 确认使用正确的 PyTorch 模型而非 ONNX

### 问题 3：认证错误
**错误**：`Authentication required`
**解决方案**：
- 设置 `HF_HUB_OFFLINE=1` 环境变量
- 在代码中使用 `use_auth_token=False` 参数
- 确保配置文件使用本地路径而非 HuggingFace 模型名

### 问题 4：内存不足或性能问题
**解决方案**：
```python
# 优化批处理大小
pipeline = Pipeline.from_pretrained("config.yaml")
pipeline.to("cuda")  # 使用 GPU 加速

# 调整批处理参数
config_with_optimized_batch = {
    "embedding_batch_size": 64,  # 根据 GPU 内存调整
    "segmentation_batch_size": 16
}
```

## pyannote.audio 3.x 版本离线使用最佳实践

### 版本选择建议
- **推荐版本**：pyannote.audio 3.1+
- **优势**：纯 PyTorch 实现，无 ONNX 依赖
- **性能**：比 2.x 版本快 2-3 倍

### 性能优化配置

```python
import torch
from pyannote.audio import Pipeline

# 加载并优化 pipeline
pipeline = Pipeline.from_pretrained("models/config.yaml")

# GPU 优化
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))
    
# 批处理优化
optimized_config = {
    "embedding_batch_size": 128,  # 根据 GPU 内存调整
    "segmentation_batch_size": 32
}
```

### 安全性最佳实践

1. **网络隔离**：确保 `HF_HUB_OFFLINE=1` 设置
2. **访问控制**：限制对模型文件的访问权限
3. **资源限制**：设置处理时间和内存限制
4. **日志记录**：实现安全的处理日志（不暴露敏感数据）

### 生产环境部署检查清单

- [ ] 模型文件命名正确（包含 `pyannote_model_` 前缀）
- [ ] 环境变量设置正确（`HF_HUB_OFFLINE=1`）
- [ ] 配置文件路径使用绝对路径
- [ ] 测试完全离线运行（断网测试）
- [ ] GPU 优化配置（如果可用）
- [ ] 错误处理和日志记录完整
- [ ] 性能基准测试完成

通过遵循这些最佳实践，您可以在完全离线的环境中成功部署和运行 pyannote.audio speaker diarization，实现高质量的语音分割功能。