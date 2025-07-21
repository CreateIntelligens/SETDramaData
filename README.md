# Breeze ASR - Speaker Diarization and Audio Segmentation

自動說話人分離和音檔切分工具，使用 pyannote 和 Docker 容器化部署。

## 功能特色

- 🎤 **說話人分離** - 使用 pyannote 進行精確的說話人識別
- ✂️ **智能切分** - 按時間範圍切分音檔（2-15秒）
- 🔗 **智能合併** - 自動合併同說話人的連續片段
- 🚫 **過濾功能** - 自動過濾空白和過短片段
- 📁 **LibriTTS 格式** - 標準的語音資料集格式
- 🐳 **Docker 支援** - 容器化部署，環境一致性
- 📊 **進度顯示** - 實時進度條和處理狀態

## 快速開始

### 方法一：Docker 部署（推薦）

```bash
# 1. 設定環境
cp .env.example .env
# 編輯 .env 添加你的 HUGGINGFACE_TOKEN

# 2. 準備資料
# 將音檔和字幕檔放入 ./data 目錄

# 3. 運行容器
./docker-run.sh

# 或使用 GPU 加速
./docker-run.sh gpu
```

### 方法二：本地安裝

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定環境變數
cp .env.example .env
# 編輯 .env 添加你的 HUGGINGFACE_TOKEN

# 3. 運行互動介面
./interactive.sh
```

## Docker 使用方法

### 基本命令

```bash
# 互動式運行
./docker-run.sh

# GPU 加速
./docker-run.sh gpu

# 使用 Docker Compose
./docker-run.sh compose

# 只建立映像
./docker-run.sh build

# 進入容器 Shell
./docker-run.sh shell
```

### 目錄結構

```
breeze_asr/
├── data/                    # 輸入資料目錄
│   └── 願望(音軌及字幕檔)/
├── output/                  # 輸出結果目錄
├── src/                     # 源代碼
├── Dockerfile              # Docker 映像配置
├── docker-compose.yml      # Docker Compose 配置
├── docker-run.sh           # Docker 運行腳本
├── interactive.sh          # 互動式介面
└── .env                    # 環境變數
```

## 使用方法

### 互動式選單

```
🎤 Breeze ASR - Speaker Diarization Tool
==========================================

請選擇功能：
1. 處理集數 (Process Episodes)
2. 處理並切分 (Process & Split)
3. 切分訓練/測試集 (Split Dataset)
4. 查看狀態 (View Status)
5. 離開 (Exit)
```

### 支援的輸入格式

- **單集**: `1`
- **多集**: `1 3 5`
- **範圍**: `2-6`

### 輸出格式

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
└── split_dataset/  # 切分後的訓練/測試集
    ├── train/
    └── test/
```

## 環境需求

### 系統需求

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 10GB+ 可用空間

### GPU 支援（可選）

- NVIDIA GPU
- NVIDIA Container Toolkit
- CUDA 11.8+

### 本地安裝需求

- Python 3.9+
- CUDA 11.8+（GPU 使用）
- FFmpeg
- libsndfile

## 環境變數

```bash
# .env 文件
HUGGINGFACE_TOKEN=your_token_here
HF_TOKEN=your_token_here

# 可選：CUDA 設定
CUDA_VISIBLE_DEVICES=0
```

## 性能優化

### GPU 使用

```bash
# 檢查 GPU 狀態
nvidia-smi

# 使用 GPU 運行
./docker-run.sh gpu
```

### 記憶體優化

- 推薦 8GB+ RAM
- 大音檔建議 16GB+ RAM
- 可調整 Docker 記憶體限制

## 故障排除

### 常見問題

1. **Docker 未運行**
   ```bash
   systemctl start docker
   ```

2. **權限問題**
   ```bash
   sudo usermod -aG docker $USER
   ```

3. **GPU 不可用**
   ```bash
   # 安裝 NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. **記憶體不足**
   - 減少並行處理數量
   - 使用較小的音檔進行測試
   - 增加系統記憶體

5. **模型下載失敗**
   - 檢查網路連接
   - 確認 HUGGINGFACE_TOKEN 有效
   - 使用 VPN（如果需要）

## 進階使用

### 自定義配置

```bash
# 修改處理參數
python src/pyannote_speaker_segmentation.py \
    audio.wav subtitle.txt \
    --min_duration 1.0 \
    --max_duration 20.0 \
    --episode_num 1
```

### 批次處理

```bash
# 處理多個集數
for i in {1..10}; do
    ./docker-run.sh shell -c "python src/pyannote_speaker_segmentation.py ..."
done
```

## 授權

本專案使用 MIT 授權。請參閱 LICENSE 文件。

## 貢獻

歡迎提交 Issue 和 Pull Request。

## 更新日誌

- v0.1.0: 初始版本
- v0.1.1: 添加 Docker 支援
- v0.1.2: 改進進度條顯示
- v0.1.3: 添加狀態管理和一條龍服務