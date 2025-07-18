# Breeze ASR - Speaker Diarization and Audio Segmentation

自動說話人分離和音檔切分工具，使用 pyannote 和 pyenv + poetry 環境管理。

## 功能特色

- 🎤 **說話人分離** - 使用 pyannote 進行精確的說話人識別
- ✂️ **智能切分** - 按時間範圍切分音檔（2-15秒）
- 🔗 **智能合併** - 自動合併同說話人的連續片段
- 🚫 **過濾功能** - 自動過濾空白和過短片段
- 📁 **自動命名** - 包含說話人、時間戳的檔名格式

## 環境需求

- Python 3.9+
- CUDA (可選，用於 GPU 加速)
- Hugging Face Token (用於 pyannote 模型)

## 快速開始

### 1. 設定環境

```bash
# 安裝依賴
pip install -r requirements.txt

# 設定 HF Token
cp .env.example .env
# 編輯 .env 文件，添加你的 HUGGINGFACE_TOKEN

# 或使用 poetry
poetry install
```

### 2. 執行分離

```bash
# 處理單集
./breeze.sh episode 1

# 處理所有集
./breeze.sh all

# 處理範圍
./breeze.sh all 1 5
```

## 使用方法

### 基本命令

```bash
./breeze.sh <命令> [參數]
```

### 可用命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `episode <num>` | 處理單集 | `./breeze.sh episode 1` |
| `all [start] [end]` | 處理全部或範圍 | `./breeze.sh all 1 5` |
| `split <method>` | 切分訓練/測試集 | `./breeze.sh split speaker` |
| `help` | 顯示幫助訊息 | `./breeze.sh help` |

### 切分方法

| 方法 | 說明 | 範例 |
|------|------|------|
| `speaker` | 按說話人切分 | `./breeze.sh split speaker` |
| `files` | 按檔案切分 | `./breeze.sh split files` |
| `episode <nums>` | 按集數切分 | `./breeze.sh split episode 2 5` |

## 輸出格式

### 檔案命名規則

```
{speaker_id}_{chapter_id}_{paragraph_id}_{sentence_id}.wav
{speaker_id}_{chapter_id}_{paragraph_id}_{sentence_id}.normalized.txt
```

範例：`000_001_000001_000001.wav`

### 輸出目錄結構

```
output/
├── 000/          # Speaker 0
│   └── 001/      # Chapter 1
│       ├── 000_001_000001_000001.wav
│       ├── 000_001_000001_000001.normalized.txt
│       └── 000_001.trans.tsv
├── 001/          # Speaker 1
│   └── 001/      # Chapter 1
│       ├── 001_001_000001_000001.wav
│       ├── 001_001_000001_000001.normalized.txt
│       └── 001_001.trans.tsv
└── 010/          # Speaker 10 (第2集)
    └── 002/      # Chapter 2
        ├── 010_002_000001_000001.wav
        ├── 010_002_000001_000001.normalized.txt
        └── 010_002.trans.tsv
```

## 環境變數

設定 Hugging Face Token：

```bash
export HUGGINGFACE_TOKEN="your_token_here"
# 或
export HF_TOKEN="your_token_here"
```

## 手動使用

如果需要手動控制環境：

```bash
# 啟動 poetry 環境
poetry shell

# 直接執行 Python 腳本
python pyannote_speaker_segmentation.py audio.wav subtitles.txt --help
```

## 故障排除

### 常見問題

1. **pyenv 未安裝**
   ```bash
   # 安裝 pyenv
   curl https://pyenv.run | bash
   ```

2. **poetry 未安裝**
   ```bash
   # 安裝 poetry
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Hugging Face Token 錯誤**
   - 前往 https://huggingface.co/settings/tokens 取得 token
   - 設定環境變數 `HUGGINGFACE_TOKEN`

4. **模型下載失敗**
   - 確認網路連接
   - 檢查 token 權限
   - 嘗試不同的模型版本

## 依賴套件

主要依賴：
- pyannote.audio >= 3.1.0
- librosa >= 0.10.0
- soundfile >= 0.12.0
- numpy >= 1.24.0
- torch >= 2.0.0

完整依賴清單請參考 `pyproject.toml`。