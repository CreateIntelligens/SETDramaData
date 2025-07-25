# PyAnnote 離線模式實作總結

## 🎯 核心問題與解決方案

### 問題背景
- **目標**: 讓 PyAnnote 在完全離線環境（H100）中運作
- **挑戰**: HuggingFace 模型預設需要網路連接
- **症狀**: `argument of type 'NoneType' is not iterable` 和 `cannot find requested files in local cache`

### 根本原因
1. **符號連結損壞**: H100 環境中的 `pytorch_model.bin` 和 `config.yaml` 變成空檔案
2. **快取結構不完整**: 傳輸過程中符號連結被破壞
3. **本地能過關**: 因為本地 Docker 有網路連接作為 fallback

## 🔧 解決方案

### 1. 自動修復功能
已整合到主程式 `src/pyannote_speaker_segmentation.py` 中：

```python
def fix_huggingface_symlinks():
    # 修復三個關鍵檔案的符號連結
    symlink_fixes = [
        {
            "model": "segmentation-3.0",
            "file": "pytorch_model.bin",
            "blob": "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
        },
        {
            "model": "wespeaker-voxceleb-resnet34-LM", 
            "file": "pytorch_model.bin",
            "blob": "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
        },
        {
            "model": "speaker-diarization-3.1",
            "file": "config.yaml",
            "blob": "5402e3ca79b6cfa5b0ec283eed920cafe45ee39b"
        }
    ]
```

### 2. 正確的載入方式
```python
# ✅ Pipeline 載入（不支援 local_files_only）
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    cache_dir=str(models_dir),
    use_auth_token=None
)

# ✅ Model 載入（支援 local_files_only）
model = Model.from_pretrained(
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    cache_dir=str(models_dir),
    local_files_only=True,
    use_auth_token=None
)
```

## 📁 關鍵檔案

### 已修改的檔案
1. **`src/pyannote_speaker_segmentation.py`** - 主程式，已整合自動修復功能
2. **`test_offline_complete.py`** - 測試腳本，已整合自動修復功能

### 測試檔案
- **`debug_diarization.py`** - 除錯工具，檢查配置檔案和模型載入
- **`fix_config.py`** - 獨立的配置檔案修復工具（備用）

## 🚀 部署流程

### 上傳到 H100
```bash
scp src/pyannote_speaker_segmentation.py user@h100:/workspace/etl/src/
scp test_offline_complete.py user@h100:/workspace/etl/
```

### 測試步驟
```bash
# 1. 測試離線載入
python test_offline_complete.py

# 2. 執行完整處理
./etl.sh
```

### 預期結果
```
🔧 檢查並修復符號連結...
✅ speaker-diarization-3.1/config.yaml: 符號連結已修復
🎉 所有測試通過！(3/3)
✅ 完全離線模式運作正常
```

## 🔍 除錯指令

### 檢查符號連結狀態
```bash
# 檢查 .bin 檔案
find models/huggingface -name "*.bin" -exec ls -lh {} \;

# 檢查 config.yaml
find models/huggingface -name "config.yaml" -exec ls -lh {} \;
```

### 檢查 blob 檔案
```bash
ls -la models/huggingface/models--pyannote--*/blobs/
```

## 💡 關鍵知識點

### 1. API 差異
- **Pipeline.from_pretrained()**: 不支援 `local_files_only` 參數
- **Model.from_pretrained()**: 支援 `local_files_only` 參數

### 2. 環境變數
```python
os.environ.update({
    'TRANSFORMERS_OFFLINE': '1',
    'HUGGINGFACE_HUB_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1'
})
```

### 3. 符號連結結構
```
models/huggingface/models--pyannote--*/snapshots/*/pytorch_model.bin
-> ../../blobs/{blob_id}
```

## ⚠️ 常見問題

### 1. `NoneType is not iterable`
- **原因**: config.yaml 是空檔案
- **解決**: 修復 config.yaml 的符號連結

### 2. `cannot find requested files in local cache`
- **原因**: pytorch_model.bin 符號連結損壞
- **解決**: 修復 pytorch_model.bin 的符號連結

### 3. `Ran out of input`
- **原因**: .bin 檔案損壞或為空
- **解決**: 重新建立正確的符號連結

## 🎉 成功標準

- ✅ 3/3 模型測試通過
- ✅ 完全不依賴網路連接
- ✅ 自動修復損壞的符號連結
- ✅ 在 H100 環境中穩定運作

## ⚠️ 最新狀況更新

### 當前問題
- **符號連結修復功能已實作** - 但 H100 環境中仍然失敗
- **config.yaml 修復已加入** - 但可能還有其他問題
- **需要進一步診斷** - 可能是更深層的配置或依賴問題

## 🔍 下一步可能方向

### 1. **檢查 H100 環境差異**
```bash
# 比較 H100 和本地的 pyannote 版本
pip list | grep pyannote
python -c "import pyannote.audio; print(pyannote.audio.__version__)"

# 檢查 Python 版本差異
python --version
```

### 2. **完全繞過 HuggingFace Hub**
- **方向**: 不使用 `Pipeline.from_pretrained()` 和 `Model.from_pretrained()`
- **策略**: 直接載入 PyTorch 模型檔案，手動建構 pipeline
- **優點**: 完全控制載入過程，不依賴 HuggingFace 的快取機制

### 3. **使用本地模型檔案**
```python
# 可能的方向：直接載入 .bin 檔案
import torch
model_path = "/workspace/etl/models/huggingface/.../pytorch_model.bin"
state_dict = torch.load(model_path, map_location=device)
```

### 4. **環境隔離測試**
- **測試**: 在 H100 建立全新的 conda 環境
- **重裝**: 重新安裝 pyannote-audio 和相關依賴
- **對比**: 確認是否是環境污染問題

### 5. **配置檔案手動修正**
- **檢查**: H100 中的 config.yaml 內容是否與本地完全一致
- **修正**: 可能需要手動調整配置中的模型路徑
- **測試**: 使用絕對路徑而非相對路徑

## 📝 下次對話重點

1. **符號連結修復已實作** - 但 H100 環境仍有問題
2. **需要更深入診斷** - 可能是版本、環境或配置問題
3. **考慮完全繞過 HuggingFace** - 直接載入 PyTorch 模型
4. **測試環境差異** - H100 vs 本地的詳細對比

**當前狀況**: 符號連結修復功能已完成，但 H100 環境中仍有未知問題需要進一步診斷和解決。
