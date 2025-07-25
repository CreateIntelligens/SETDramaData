# 🎯 使用說明 - 完全離線的語者識別系統

## ⚡ 快速開始

### 1️⃣ 設定 HuggingFace 快取（只需執行一次）
```bash
python setup_hf_cache.py
```

### 2️⃣ 執行語者識別
```bash
export HUGGINGFACE_HUB_OFFLINE=1
python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1
```

## 📁 核心檔案說明

- `setup_hf_cache.py` - 設定 HuggingFace 官方離線快取結構
- `src/pyannote_speaker_segmentation.py` - 主程式

## 🔧 setup_hf_cache.py 做了什麼？

1. **建立標準 HF 快取結構**：
   ```
   models/huggingface/hub/
   ├── models--pyannote--segmentation-3.0/
   ├── models--pyannote--wespeaker-voxceleb-resnet34-LM/
   └── models--pyannote--speaker-diarization-3.1/
   ```

2. **修正配置檔案**：
   - 將相對路徑改為標準 repo ID
   - 避免 HuggingFace Hub 路徑驗證錯誤

3. **完全離線運作**：
   - 使用 HuggingFace 官方推薦方式
   - 不需要網路連接

## ✅ 執行結果

執行 `setup_hf_cache.py` 後會看到：
```
🚀 設定 HuggingFace 官方推薦的離線快取結構
🔄 設定 pyannote/segmentation-3.0...
🔄 設定 pyannote/wespeaker-voxceleb-resnet34-LM...
🔄 設定 pyannote/speaker-diarization-3.1...
📝 建立修正後的 config.yaml...
✅ 完成！
```

然後主程式就能完全離線運作了！

## 🎉 優勢

- ✅ **官方推薦方式** - 使用 HuggingFace 標準做法
- ✅ **完全離線** - 不需要任何網路連接  
- ✅ **簡單易用** - 只需執行兩個命令
- ✅ **避免錯誤** - 解決路徑驗證問題