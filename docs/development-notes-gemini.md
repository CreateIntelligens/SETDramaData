ffline/cache/hub"
```
[Hugging Face](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables) [GitHub](https://github.com/huggingface/transformers/issues/30345)

### Python 代码实现

```python
import os
from pathlib import Path
from pyannote.audio import Pipeline

# 设置离线环境
os.envi# Gemini 開發與維護日誌

本檔案記錄了 Gemini 在維護此專案時所做的變更、遇到的問題以及對程式碼的觀察與建議。

## 日期：2025年7月22日

### 核心問題：處理短音檔時，Embedding 提取失敗

使用者回報，在處理音訊片段時，會出現 `無有效embedding` 以及後續的 `'dict' object has no attribute 'dim'` 錯誤，導致處理流程中斷。

### 除錯與修復過程

1.  **初步分析**：一開始懷疑是 `pyannote` 模型無法處理剛好 2.0 秒的邊界情況，嘗試在 `compute_average_embedding` 函式中對短音訊進行填充 (padding)，但未解決問題。

2.  **發現編碼錯誤**：在試圖重現問題時，發現 Windows 環境下會發生 `UnicodeEncodeError: 'cp950' codec can't encode character` 的錯誤。這是因為 Python 腳本中的 Emoji 圖示無法被系統預設的 `cp950` 編碼處理。
    *   **修復**：在 `src/process_utils.sh` 中，執行 Python 腳本的指令前，加上 `PYTHONIOENCODING=UTF-8` 環境變數，強制使用 UTF-8 編碼，成功解決了日誌輸出的問題。

3.  **定位根本原因**：在解決了編碼問題後，`'dict' object has no attribute 'dim'` 的錯誤日誌才真正顯現。經過追查，發現問題的根源並不在計算全域 embedding 的 `compute_average_embedding` 函式，而是在「傳統模式」下，用於**驗證是否要合併連續片段**的 `verify_speaker_similarity` 函式中。
    *   **錯誤點**：`main` 函式中，傳遞給 `merge_consecutive_segments_with_verification` 的是 `embedding_inference.model`，這是一個底層的 `pyannote.audio.Model` 物件。然而，在 `verify_speaker_similarity` 函式內部，卻把它當作高階的 `pyannote.audio.Inference` 物件來使用（傳遞了一個包含檔案路徑的字典給它）。`Model` 物件預期接收的是一個音訊波形張量 (Tensor)，而不是字典，因此拋出了 `'dict' object has no attribute 'dim'` 的錯誤。

4.  **最終修復方案**：
    *   **重構 `verify_speaker_similarity`**：將其修改為直接接收**音訊波形陣列 (numpy array)** 和取樣率，而不是檔案路徑。函式內部會正確地將波形轉換為 PyTorch 張量，並傳遞給 `Model` 物件。
    *   **重構 `merge_consecutive_segments_with_verification`**：修改此函式，讓它在進入迴圈前**只載入一次**完整的音訊檔案。然後在迴圈中，將載入的波形資料傳遞給 `verify_speaker_similarity` 進行驗證。這不僅修正了錯誤，也大幅提升了驗證流程的效率，避免了重複的檔案 I/O。
    *   **同步改進 `compute_average_embedding`**：雖然此函式不是這次錯誤的直接原因，但為了程式碼的穩健性和一致性，也將其修改為直接處理音訊波形，而不是傳遞檔案路徑。

### 總結

這次的修復不僅解決了使用者回報的 embedding 提取失敗問題，也一併處理了 Windows 環境下的編碼錯誤，並對音訊處理流程進行了效率最佳化。

### 對專案的觀察與未來改進建議

1.  **Embedding 提取邏輯可以統一**：`compute_average_embedding` 和 `verify_speaker_similarity` 中都有提取 embedding 的邏輯。未來可以將其抽象成一個更通用的工具函式，例如 `extract_embedding(waveform, model)`，以減少程式碼重複。

2.  **`Model` vs `Inference` 物件的使用**：腳本中對於 `pyannote` 的 `Model` 和 `Inference` 物件的使用有些混亂，這是導致這次錯誤的根本原因。建議未來在傳遞參數時，能更明確地標示其類型，或在函式內部進行檢查，以增加程式碼的穩健性。

3.  **設定管理**：目前許多設定是透過 shell 腳本的環境變數傳遞給 Python。當設定變得更複雜時，可以考慮使用一個統一的設定檔（如 `config.yaml`），讓 Python 腳本直接讀取，以簡化 shell 腳本的邏輯。
