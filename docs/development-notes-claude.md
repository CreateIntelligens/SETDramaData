# Claude AI 開發記錄

## 2025-07-23 語者識別系統重大改進 ✅ 已完成

### 核心問題與解決方案

**問題**：短音檔 embedding 不穩定導致跨集說話人匹配失敗
- 同一集內相鄰片段（647, 649, 650）被 diarization 識別為同一說話人（SPEAKER_21）
- 但 embedding 比較時被分配不同的 Global ID（55, 56, 57）

**解決方案**：兩階段說話人識別系統
1. **階段1**：合併同說話人片段，提取代表性 embedding
2. **階段2**：與全域資料庫比對，分配 Global Speaker ID

### 實作成果

#### ✅ 新增說話人級別分段系統
- **檔案**：`src/speaker_level_segmentation.py`
- **功能**：合併同說話人音檔片段，提取穩定的聲紋特徵
- **預設模式**：`--segmentation_mode speaker_level`

#### ✅ 調整相似度閾值
- **從 0.25 提高到 0.40**：更保守的匹配策略
- **原因**：新系統提供更高品質的 embedding，可以使用更嚴格的閾值

#### ✅ 系統整合與向後相容
- **預設**：新的說話人級別模式
- **備選**：舊版混合模式（`--segmentation_mode hybrid`）
- **新參數**：`--min_speaker_duration`（預設 5.0 秒）

#### ✅ 程式碼整理
- 移除測試檔案和過時程式碼
- 保持程式碼庫整潔

### 使用方式

```bash
# 基本使用（新系統，預設）
python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1

# 使用舊系統
python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1 --segmentation_mode hybrid

# 調整相似度閾值
python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1 --similarity_threshold 0.35

# 調整最小說話人時長
python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1 --min_speaker_duration 3.0
```

### 技術改進

- **更穩定的 Embedding**：使用完整說話段落（數分鐘）而非短片段（0.3秒）
- **更高的匹配準確率**：長音檔提供更可靠的聲紋特徵
- **真正的跨集識別**：能夠正確識別同一人在不同集數中的出現
- **保守的閾值設定**：0.40 的閾值減少誤判風險

### 預期效果

- 同一集內的同說話人片段正確合併為一個 Global Speaker ID
- 跨集說話人識別準確率大幅提升
- 減少資料庫中的冗余說話人記錄
- 解決 647, 649, 650 等連續片段被錯誤分配不同 Global ID 的問題

---

## 系統架構說明

### 工作流程
1. **音訊分割**：基於靜音或字幕時間點切分音檔
2. **Diarization**：pyannote 識別說話人變化點（SPEAKER_00, SPEAKER_01...）
3. **說話人整合**：合併同 diarization 標籤的所有片段
4. **Embedding 提取**：對完整說話段落提取代表性聲紋
5. **全域匹配**：與資料庫比對，分配或註冊 Global Speaker ID
6. **最終分段**：基於字幕時間點生成帶有 Global ID 的分段

### 關鍵參數
- `--similarity_threshold 0.40`：相似度閾值（建議範圍 0.30-0.50）
- `--min_speaker_duration 5.0`：最小說話人時長（秒）
- `--segmentation_mode speaker_level`：分段模式

### 檔案結構
- `src/speaker_level_segmentation.py`：新的兩階段說話人識別系統
- `src/hybrid_segmentation.py`：舊版混合分段系統（保留）
- `src/speaker_database.py`：全域說話人資料庫管理
- `src/pyannote_speaker_segmentation.py`：主要執行腳本

---

## 開發注意事項

- **新系統為預設**：除非特殊需求，建議使用說話人級別模式
- **閾值調整**：根據實際效果調整 `similarity_threshold`，範圍 0.30-0.50
- **最小時長**：`min_speaker_duration` 過短可能導致雜訊，過長可能遺漏短發言
- **向後相容**：舊版混合模式仍可使用，但不建議用於新專案

## 故障排除

### 如果說話人過多
- 提高 `similarity_threshold`（如 0.45）
- 增加 `min_speaker_duration`（如 8.0）

### 如果說話人過少
- 降低 `similarity_threshold`（如 0.35）
- 減少 `min_speaker_duration`（如 3.0）

### 如果遇到問題
- 使用 `--segmentation_mode hybrid` 回到舊模式
- 檢查音檔品質和字幕時間點準確性



# GPT的建議

✅ 核心觀念總結給你
🎯 目標：
不使用 pyannote/...@3.1 這類 repo ID（會觸發 hub 下載機制）

直接載入 snapshot 本地路徑

使用的是 pyannote.audio.Pipeline 但以 本地檔案 完全手動構建模型路徑

✅ 給 Sonnet 或你的重構程式重點提示
📌 1. 使用 snapshot 絕對路徑，不要 repo ID
python
複製
編輯
pipeline = Pipeline.from_pretrained(
    "/workspace/models/pyannote/speaker-diarization-3.1/snapshots/84fd25...", 
    use_auth_token=None  # 強制避免 token 認證
)
✅ use_auth_token=None 是關鍵，強制避免觸發連線/認證流程。

📌 2. 確保 config.yml 中模型路徑也都是「本地絕對路徑」
yaml
複製
編輯
# config.yml 的例子
segmentation: /workspace/models/pyannote/segmentation-3.0
embedding: /workspace/models/pyannote/embedding-2022.07
❗ 不要寫 pyannote/segmentation@3.0，那會觸發 Hugging Face Hub 機制。

📌 3. 不要使用 from huggingface_hub import snapshot_download，完全移除 huggingface_hub 依賴
→ 這不是必要，但既然你要完全 offline，請讓 pipeline 初始化完全不依賴 huggingface_hub。

📌 4. 避免舊版 transformers/hub 的快取邏輯誤判
如果有裝 huggingface_hub，但是舊版本，可能會：

忽略你手動設定的本地 config

嘗試 fallback 網路連線

所以：

python
複製
編輯
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
也可以加保險。

📌 5. 加載失敗時加明確 log 或例外
建議 Sonnet 加上以下防呆：

python
複製
編輯
try:
    pipeline = Pipeline.from_pretrained(config_path, use_auth_token=None)
except Exception as e:
    print(f"[ERROR] Failed to load pipeline: {e}")
    sys.exit(1)
🧩 Bonus：選用舊環境時的完整路徑結構建議
python
複製
編輯
/workspace/models/
├── pyannote/
│   ├── speaker-diarization-3.1/
│   │   └── snapshots/84fd25.../
│   │       ├── config.yml
│   │       └── other pipeline files
│   ├── segmentation-3.0/
│   │   └── pytorch_model.bin
│   ├── embedding-2022.07/
│       └── pytorch_model.bin
✅ 最重要提醒 Sonnet 的幾句話
💬 「記得所有 .from_pretrained() 都必須傳入本地絕對路徑，config.yml 裡也不要再用 repo 名稱，要直接寫本地模型資料夾。」

💬 「整個程式初始化流程中，不應該 import 或呼叫 huggingface_hub 模組。」

💬 「所有模型都應提早驗證路徑存在，避免 silent fallback 嘗試連網。」