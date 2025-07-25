# CosyVoice 2.0 數據格式規範

## 概述

本文檔基於對 LibriTTS 測試數據的實際分析，總結了 CosyVoice 2.0 訓練所需的標準數據格式。

## 原始數據格式 (以 LibriTTS 為例)

### 目錄結構
```
LibriTTS/
├── test-clean/
│   ├── 1089/                    # 說話人ID
│   │   ├── 134686/              # 章節ID  
│   │   │   ├── 1089_134686_000001_000001.wav           # 音頻文件
│   │   │   ├── 1089_134686_000001_000001.normalized.txt # 標準化文本
│   │   │   ├── 1089_134686_000001_000001.original.txt   # 原始文本
│   │   │   ├── 1089_134686_000002_000000.wav
│   │   │   ├── 1089_134686_000002_000000.normalized.txt
│   │   │   └── ...
│   │   └── 134691/
│   ├── 1188/
│   └── ...
```

### 文件命名規則
- **音頻文件**: `{說話人ID}_{章節ID}_{語音序號}_{片段序號}.wav`
- **文本文件**: `{說話人ID}_{章節ID}_{語音序號}_{片段序號}.normalized.txt`

**實際範例**:
- `1089_134686_000001_000001.wav`
- `1089_134686_000001_000001.normalized.txt`

### 文本內容格式
**normalized.txt 內容範例**:
```
He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce. Stuff it into you, his belly counselled him.
```

**特點**:
- 單行文本，無換行符
- 已經進行了文本標準化處理
- 標點符號保留
- 大小寫規範化

## 處理後的 Kaldi 格式

經過 `prepare_data.py` 處理後，生成以下 4 個標準文件：

### 1. wav.scp (音頻文件路徑映射)
**格式**: `語音ID 音頻文件絕對路徑`

**實際內容範例**:
```
1089_134686_000001_000001 /workspace/CosyVoice/data/libritts/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav
1089_134686_000002_000000 /workspace/CosyVoice/data/libritts/LibriTTS/test-clean/1089/134686/1089_134686_000002_000000.wav
1089_134686_000002_000001 /workspace/CosyVoice/data/libritts/LibriTTS/test-clean/1089/134686/1089_134686_000002_000001.wav
```

### 2. text (文本轉錄內容)
**格式**: `語音ID 轉錄文本內容`

**實際內容範例**:
```
1089_134686_000001_000001 He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce. Stuff it into you, his belly counselled him.
1089_134686_000002_000000 It would be a gloomy secret night.
1089_134686_000002_000001 After early nightfall the yellow lamps would light up, here and there, the squalid quarter of the brothels.
```

### 3. utt2spk (語音到說話人映射)
**格式**: `語音ID 說話人ID`

**實際內容範例**:
```
1089_134686_000001_000001 1089
1089_134686_000002_000000 1089
1089_134686_000002_000001 1089
1089_134686_000002_000002 1089
```

### 4. spk2utt (說話人到語音列表映射)
**格式**: `說話人ID 語音ID1 語音ID2 語音ID3 ...`

**實際內容範例**:
```
1089 1089_134686_000001_000001 1089_134686_000002_000000 1089_134686_000002_000001 1089_134686_000002_000002 1089_134686_000002_000003 ...
1188 1188_133604_000000_000000 1188_133604_000004_000005 1188_133604_000006_000000 ...
```

## 數據處理腳本分析

### prepare_data.py 核心邏輯
```python
# 關鍵處理步驟：
1. 遍歷所有 .wav 文件: glob.glob('{}/*/*/*wav'.format(args.src_dir))
2. 尋找對應文本文件: wav.replace('.wav', '.normalized.txt')  
3. 提取說話人ID: utt.split('_')[0]
4. 生成映射關係並寫入 4 個文件
```

### 命名提取規則
- **語音ID**: 文件名去除 `.wav` 後綴
- **說話人ID**: 語音ID 按 `_` 分割後的第一部分
- **文本內容**: 從 `.normalized.txt` 文件讀取，去除換行符

## 後續處理步驟

### 特徵提取
處理完成後需要進一步提取特徵：

1. **說話人嵌入提取**:
   ```bash
   tools/extract_embedding.py --dir data/custom --onnx_path campplus.onnx
   ```
   生成: `utt2embedding.pt`, `spk2embedding.pt`

2. **語音 Token 提取**:
   ```bash
   tools/extract_speech_token.py --dir data/custom --onnx_path speech_tokenizer_v2.onnx
   ```
   生成: `utt2speech_token.pt`

3. **Parquet 格式轉換**:
   ```bash
   tools/make_parquet_list.py --src_dir data/custom --des_dir data/custom/parquet
   ```
   生成: `parquet/` 目錄及相關文件

### 最終數據目錄結構
```
data/custom/
├── wav.scp                  # 音頻路徑映射
├── text                     # 文本轉錄
├── utt2spk                  # 語音->說話人映射
├── spk2utt                  # 說話人->語音映射  
├── utt2embedding.pt         # 語音嵌入特徵
├── spk2embedding.pt         # 說話人嵌入特徵
├── utt2speech_token.pt      # 語音 Token
└── parquet/                 # 訓練用 Parquet 格式
    ├── data.list
    ├── shard_0000.parquet
    └── shard_0001.parquet
```

## 實際統計數據 (test-clean 數據集)

基於實際處理的 LibriTTS test-clean 數據集：

- **總文件數**: 10,571 個語音文件
- **說話人數**: 40 位說話人
- **文件大小**:
  - wav.scp: 582KB
  - text: 630KB  
  - utt2spk: 145KB
  - spk2utt: 122KB

## 適配自定義數據的要求

基於以上分析，自定義數據需要滿足：

1. **文件結構**: 清晰的說話人/語音組織方式
2. **命名規範**: 能夠從文件名提取說話人ID
3. **文本對應**: 每個音頻文件有對應的文本轉錄
4. **編碼格式**: UTF-8 文本編碼
5. **音頻格式**: WAV 格式音頻文件

## 注意事項

1. **路徑問題**: 所有路徑在 wav.scp 中必須是絕對路徑
2. **編碼問題**: 文本文件必須使用 UTF-8 編碼
3. **映射完整性**: 四個文件中的語音ID必須完全一致
4. **說話人ID**: 建議使用數字或簡單字母組合，避免特殊字符

這個格式規範基於對實際運行的 LibriTTS 數據的分析，可以作為準備自定義數據的參考標準。