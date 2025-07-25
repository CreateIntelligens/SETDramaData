# SQLite資料庫遷移指南

## 概述

系統已從JSON檔案狀態管理升級為SQLite資料庫，提供更好的效能和資料管理功能。

## 主要改進

### 🚀 效能提升
- 快速相似度查詢和索引
- 大量speaker時記憶體使用更有效率
- 並發存取安全

### 🗄️ 資料結構化
- 標準化的資料庫架構
- 支援複雜查詢和統計
- 自動資料完整性檢查

### 🛠️ 管理工具
- 豐富的CLI管理工具
- 互動式選單整合
- 匯出/匯入功能

## 自動遷移

系統會自動檢測並遷移現有的`processing_state.json`：

1. **執行處理時自動遷移**：
   ```bash
   python src/pyannote_speaker_segmentation.py audio.wav subtitle.txt --episode_num 1
   ```

2. **手動遷移**：
   ```bash
   python src/speaker_db_manager.py migrate processing_state.json
   ```

3. **透過互動選單**：
   ```bash
   ./interactive.sh
   # 選擇: 8. 資料庫管理 > 6. 從JSON遷移
   ```

## 新功能

### 1. 資料庫管理選單
- 查看資料庫統計
- 列出所有speaker
- 查看speaker詳細資訊
- 查看集數speaker對應
- 匯出/備份資料庫

### 2. CLI管理工具
```bash
# 查看統計
python src/speaker_db_manager.py stats

# 列出所有speaker
python src/speaker_db_manager.py list

# 查看特定speaker
python src/speaker_db_manager.py speaker 0

# 查看集數對應
python src/speaker_db_manager.py episode 1

# 匯出備份
python src/speaker_db_manager.py export backup.json

# 建立資料庫備份
python src/speaker_db_manager.py backup
```

## 檔案結構變化

### 新增檔案
```
src/
├── speaker_database.py      # SQLite資料庫管理模組
└── speaker_db_manager.py    # CLI管理工具

speakers.db                  # SQLite資料庫檔案 (取代processing_state.json)
```

### 遷移後的檔案
```
processing_state.json        # 自動重命名為 processing_state.json.backup
```

## 資料庫架構

### speakers表
- `speaker_id`: 全域speaker ID
- `embedding`: speaker embedding (binary)
- `embedding_dim`: embedding維度
- `created_at`: 建立時間
- `updated_at`: 更新時間
- `episode_count`: 出現集數
- `segment_count`: 總segment數
- `notes`: 備註

### speaker_episodes表
- `speaker_id`: speaker ID
- `episode_num`: 集數
- `local_label`: 本地標籤 (如SPEAKER_00)
- `segment_count`: 該集的segment數
- `created_at`: 建立時間

### processing_state表
- `key`: 設定鍵值
- `value`: 設定值 (JSON格式)
- `updated_at`: 更新時間

## 離線部署支援

SQLite完全支援離線環境：
- 無需網路連線
- 所有資料存在本地檔案
- 與Docker環境完全相容

## 故障排除

### 遷移失敗
如果遷移失敗，可以：
1. 檢查JSON檔案格式
2. 確保有寫入權限
3. 手動刪除不完整的speakers.db後重試

### 資料庫損壞
```bash
# 從備份恢復
cp speakers_backup_YYYYMMDD_HHMMSS.db speakers.db

# 或從JSON重新遷移
rm speakers.db
python src/speaker_db_manager.py migrate processing_state.json.backup
```

### 相容性檢查
```bash
# 測試資料庫
python src/speaker_db_manager.py stats

# 比較遷移前後
python src/speaker_db_manager.py export current_state.json
# 比較 current_state.json 和 processing_state.json.backup
```

## 最佳實踐

1. **定期備份**：
   ```bash
   python src/speaker_db_manager.py backup
   ```

2. **監控資料庫大小**：
   ```bash
   python src/speaker_db_manager.py stats
   ```

3. **匯出重要資料**：
   ```bash
   python src/speaker_db_manager.py export important_backup.json
   ```

4. **檢查資料一致性**：
   - 定期查看speaker統計
   - 驗證集數對應關係

## 回退方案

如需回退到JSON格式：
1. 匯出資料庫：`python src/speaker_db_manager.py export rollback.json`
2. 恢復舊版程式碼
3. 將匯出的JSON重命名為`processing_state.json`

注意：回退後會失去SQLite的效能優勢和新功能。