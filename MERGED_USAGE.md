# 合併環境使用說明

基於 CosyVoice 環境，添加 etl 功能的統一容器。

## 構建方式

```bash
# 使用合併的配置構建
docker compose -f docker-compose.merged.yml build
```

## 使用方式

### 1. 使用 CosyVoice（原功能保持不變）

```bash
# 進入 CosyVoice 工作目錄，使用原有功能
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice bash

# 在容器內
conda activate cosyvoice  # 如果需要
cd /workspace/CosyVoice
# 使用 CosyVoice 的正常命令...
```

### 2. 使用 etl（新添加功能）

```bash
# 直接運行 etl（conda 環境自動激活）
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice \
    python /workspace/etl/src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1

# 進入交互模式
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice \
    bash /workspace/etl/interactive.sh

# 進入 shell（環境自動激活）
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice bash
```

## 環境特性

### 包含的功能
- ✅ **CosyVoice**: 完整的 TTS 功能
- ✅ **etl**: Drama 語音處理功能
- ✅ **共享 GPU**: 兩個專案都能使用 GPU 加速
- ✅ **優化配置**: 解決 MKL primitive 問題

### 目錄結構
```
/workspace/
├── CosyVoice/          # CosyVoice 專案（主環境）
├── etl/       # etl 專案（掛載）
├── data/               # 共享數據目錄
└── output/             # 共享輸出目錄
```

### Python 環境
- **Conda 環境**: `cosyvoice` (Python 3.10)
- **路徑配置**: 兩個專案都在 PYTHONPATH 中
- **依賴管理**: etl 依賴已安裝到 cosyvoice 環境

## 快速測試

### 測試 GPU 可用性
```bash
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice bash -c "
conda activate cosyvoice
python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
"
```

### 測試 etl 導入
```bash
docker compose -f docker-compose.merged.yml run --rm cosyvoice-setvoice bash -c "
conda activate cosyvoice
cd /workspace/etl
python -c 'from pyannote.audio import Pipeline; print(\"✅ pyannote 可用\")'
"
```

## 注意事項

1. **環境激活**: 記得在容器內使用 `conda activate cosyvoice`
2. **工作目錄**: CosyVoice 功能在 `/workspace/CosyVoice`，etl 在 `/workspace/etl`
3. **數據共享**: 兩個專案可以共享 `/workspace/data` 和 `/workspace/output` 目錄
4. **用戶權限**: 容器內使用 `louis` 用戶運行，避免權限問題

## 故障排除

如果遇到問題：
1. 確保使用正確的 conda 環境：`conda activate cosyvoice`
2. 檢查 PYTHONPATH 是否包含項目路徑
3. 驗證 GPU 是否正確掛載
4. 檢查目錄掛載是否正確