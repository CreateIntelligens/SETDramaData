# SETVoicePrep - Drama Voice Processing Toolkit

SET Drama 聲音處理工具，為 TTS 訓練資料做說話人識別與分段處理。

## 使用方法

### Docker Compose（推薦）

```bash
# 構建並運行
docker compose build
docker compose run --rm setvoiceprep bash interactive.sh

# 測試 GPU
docker compose run --rm setvoiceprep python test_gpu.py
```

### 本地安裝

```bash
# 安裝依賴（Poetry）
poetry install

# 或使用 pip
pip install -r requirements.txt

# 運行
./interactive.sh
```

## 功能

- 跨集說話人識別與匹配
- 基於字幕的音檔分段
- LibriTTS 格式輸出
- GPU 加速支援

