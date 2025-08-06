#!/bin/bash

# 智慧一條龍處理服務
# 自動檢測、清理、處理、切分

# Load utility functions
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/cleanup_utils.sh"
source "$(dirname "${BASH_SOURCE[0]}")/timing_log.sh"

# 智慧處理單集（僅處理，不切分）
smart_process_episode_only() {
    local episode_num="$1"
    
    if [ -z "$episode_num" ]; then
        echo "❌ 請提供集數"
        echo "用法: smart_process_episode_only <集數>"
        return 1
    fi
    
    echo ""
    echo "🚀 智慧一條龍處理 - 第 $episode_num 集"
    echo "=================================="
    
    # 使用 .env 的預設路徑
    local input_dir="${DEFAULT_INPUT_DIR:-data/願望(音軌及字幕檔)}"
    local output_dir="${DEFAULT_PROCESSED_DIR:-data/output}"
    local split_dir="${DEFAULT_SPLIT_DIR:-data/split_dataset}"
    
    echo "📁 使用設定路徑："
    echo "  輸入: $input_dir"
    echo "  輸出: $output_dir" 
    echo "  切分: $split_dir"
    echo ""
    
    # 1. 檢查該集是否已處理過
    local db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
    if [ -f "$db_path" ]; then
        local python_cmd=$(detect_python)
        if [ -n "$python_cmd" ]; then
            echo "🔍 檢查集數 $episode_num 處理狀態..."
            local is_processed=$($python_cmd -c "
import sys
sys.path.append('src')
from speaker_database import SpeakerDatabase
db = SpeakerDatabase()
processed = db.get_processed_episodes()
print('yes' if $episode_num in processed else 'no')
" 2>/dev/null)
            
            if [ "$is_processed" = "yes" ]; then
                echo "⚠️  集數 $episode_num 已處理過，自動清理重做..."
                
                # 自動清理該集（無需確認）
                echo "🗑️ 清理第 $episode_num 集的舊資料..."
                
                # 清理輸出檔案
                local episode_padded=$(printf "%03d" "$episode_num")
                for speaker_dir in $output_dir/*/; do
                    if [ -d "$speaker_dir" ]; then
                        local episode_dir="$speaker_dir$episode_padded"
                        if [ -d "$episode_dir" ]; then
                            rm -rf "$episode_dir"
                            echo "  ✅ 清除 $(basename "$speaker_dir")/$episode_padded"
                        fi
                    fi
                done
                
                # 清理切分資料集
                if [ -d "$split_dir" ]; then
                    find "$split_dir/train" -path "*/$episode_padded" -type d -exec rm -rf {} + 2>/dev/null
                    find "$split_dir/test" -path "*/$episode_padded" -type d -exec rm -rf {} + 2>/dev/null
                    echo "  ✅ 清除切分資料集中的相關檔案"
                fi
                
                # 更新資料庫狀態
                $python_cmd "src/database_cleanup.py" remove "$episode_num" 2>/dev/null
                echo "  ✅ 更新資料庫狀態"
                echo ""
            fi
        fi
    fi
    
    # 2. 處理該集
    echo "🎵 開始處理第 $episode_num 集..."
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    # 尋找音檔和字幕
    local episode_pattern
    if [ ${#episode_num} -eq 1 ]; then
        episode_pattern="第0${episode_num}集"
        subtitle_pattern="願望-00${episode_num}.txt"
    elif [ ${#episode_num} -eq 2 ]; then
        episode_pattern="第${episode_num}集"
        subtitle_pattern="願望-0${episode_num}.txt"
    else
        episode_pattern="第${episode_num}集"
        subtitle_pattern="願望-${episode_num}.txt"
    fi
    
    # 優先尋找 back_left.wav，其次是其他音檔
    local audio_file=$(find "$input_dir" -path "*${episode_pattern}*" -name "back_left.wav" | head -1)
    if [ -z "$audio_file" ]; then
        audio_file=$(find "$input_dir" -path "*${episode_pattern}*" \( -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" \) | head -1)
    fi
    local subtitle_file=$(find "$input_dir" -name "$subtitle_pattern" | head -1)
    
    if [ -z "$audio_file" ]; then
        echo "❌ 找不到第 $episode_num 集的音檔"
        return 1
    fi
    
    if [ -z "$subtitle_file" ]; then
        echo "❌ 找不到第 $episode_num 集的字幕檔案: $subtitle_pattern"
        return 1
    fi
    
    echo "🎵 音檔: $(basename "$audio_file")"
    echo "📝 字幕: $(basename "$subtitle_file")"
    echo ""
    
    # 檢查是否啟用 UVR5 去背
    local enable_uvr5="${ENABLE_UVR5_SEPARATION:-false}"
    local processed_audio_file="$audio_file"
    
    if [ "$enable_uvr5" = "true" ]; then
        echo "🎵 執行 UVR5 音頻去背..."
        
        local uvr5_output_dir="${UVR5_OUTPUT_DIR:-data/separated_vocals}"
        local uvr5_model="${UVR5_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
        
        # 創建 UVR5 輸出目錄
        mkdir -p "$uvr5_output_dir"
        
        # 執行 UVR5 去背
        if $python_cmd -c "
import sys
sys.path.append('src')
from uvr5_vocal_separator import create_vocal_separator
import os

separator = create_vocal_separator(
    models_dir='${UVR5_MODELS_DIR:-models/uvr5}',
    output_dir='$uvr5_output_dir',
    use_gpu=True
)

try:
    separator.initialize_separator('$uvr5_model')
    result = separator.separate_vocals('$audio_file', 'episode_${episode_num}')
    
    if result['success']:
        vocals_file = result['output_files'].get('vocals')
        if vocals_file and os.path.exists(vocals_file):
            print(f'SUCCESS:{vocals_file}')
        else:
            print('ERROR:人聲文件生成失敗')
    else:
        print(f'ERROR:{result.get(\"error\", \"未知錯誤\")}')
finally:
    separator.cleanup()
" 2>/dev/null; then
            # 解析結果
            local uvr5_output=$(${python_cmd} -c "
import sys
sys.path.append('src')
from uvr5_vocal_separator import create_vocal_separator
import os

separator = create_vocal_separator(
    models_dir='${UVR5_MODELS_DIR:-models/uvr5}',
    output_dir='$uvr5_output_dir',
    use_gpu=True
)

try:
    separator.initialize_separator('$uvr5_model')
    result = separator.separate_vocals('$audio_file', 'episode_${episode_num}')
    
    if result['success']:
        vocals_file = result['output_files'].get('vocals')
        if vocals_file and os.path.exists(vocals_file):
            print(f'SUCCESS:{vocals_file}')
        else:
            print('ERROR:人聲文件生成失敗')
    else:
        print(f'ERROR:{result.get(\"error\", \"未知錯誤\")}')
finally:
    separator.cleanup()
" 2>/dev/null)
            
            if [[ "$uvr5_output" == SUCCESS:* ]]; then
                processed_audio_file="${uvr5_output#SUCCESS:}"
                echo "✅ UVR5 去背完成: $(basename "$processed_audio_file")"
            else
                echo "❌ UVR5 去背失敗: ${uvr5_output#ERROR:}"
                echo "⚠️  將使用原始音檔繼續處理"
                processed_audio_file="$audio_file"
            fi
        else
            echo "❌ UVR5 去背執行失敗"
            echo "⚠️  將使用原始音檔繼續處理"
            processed_audio_file="$audio_file"
        fi
        echo ""
    fi
    
    # 執行 pyannote 處理（使用 .env 中的參數）
    echo "🚀 執行 pyannote 處理..."
    echo "🎵 使用音檔: $(basename "$processed_audio_file")"
    if $python_cmd "src/pyannote_speaker_segmentation.py" \
        "$processed_audio_file" "$subtitle_file" \
        --episode_num "$episode_num" \
        --output_dir "$output_dir"; then
        echo "✅ 處理完成"
    else
        echo "❌ 處理失敗"
        return 1
    fi
    
    echo ""
    echo "✅ 第 $episode_num 集處理完成！"
    
    return 0
}

# 智慧處理單集（包含切分）
smart_process_episode() {
    local episode_num="$1"
    
    if [ -z "$episode_num" ]; then
        echo "❌ 請提供集數"
        echo "用法: smart_process_episode <集數>"
        return 1
    fi
    
    # 先執行處理
    if ! smart_process_episode_only "$episode_num"; then
        echo "❌ 第 $episode_num 集處理失敗"
        return 1
    fi
    
    # 執行切分
    echo ""
    echo "📊 開始切分第 $episode_num 集..."
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    local output_dir="${DEFAULT_PROCESSED_DIR:-data/output}"
    local split_dir="${DEFAULT_SPLIT_DIR:-data/split_dataset}"
    local test_ratio="${DEFAULT_TEST_RATIO:-0.2}"
    
    if $python_cmd "src/split_dataset.py" \
        --processed_dir "$output_dir" \
        --split_dir "$split_dir" \
        --method "episode" \
        --episode_num "$episode_num" \
        --test_ratio "$test_ratio"; then
        echo "✅ 切分完成"
    else
        echo "❌ 切分失敗"
        return 1
    fi
    
    echo ""
    echo "✅ 第 $episode_num 集完整處理（包含切分）完成！"
    
    return 0
}

# 智慧處理單集 - UVR5 增強版本 (先 UVR5 再 pyannote 再切分)
smart_process_episode_with_uvr5() {
    local episode_num="$1"
    
    if [ -z "$episode_num" ]; then
        echo "❌ 請提供集數"
        echo "用法: smart_process_episode_with_uvr5 <集數>"
        return 1
    fi
    
    echo ""
    echo "🚀 UVR5 增強智慧處理 - 第 $episode_num 集"
    echo "========================================="
    echo "💡 流程：UVR5 人聲分離 → pyannote 處理 → 切分資料集"
    echo ""
    
    # 初始化時間統計日誌
    init_timing_log
    
    # 使用自定義路徑或預設路徑
    local input_dir="${CUSTOM_INPUT_DIR:-${DEFAULT_INPUT_DIR:-data/願望(音軌及字幕檔)}}"
    local uvr5_output_dir="${CUSTOM_UVR5_OUTPUT_DIR:-${UVR5_OUTPUT_DIR:-data/uvr5_separated}}"
    local pyannote_output_dir="${CUSTOM_PYANNOTE_OUTPUT_DIR:-${DEFAULT_PROCESSED_DIR:-data/output}}"
    local split_dir="${CUSTOM_SPLIT_DIR:-${DEFAULT_SPLIT_DIR:-data/split_dataset}}"
    
    echo "📁 使用路徑："
    echo "  📥 輸入: $input_dir"
    echo "  🎵 UVR5 輸出: $uvr5_output_dir"
    echo "  🤖 pyannote 輸出: $pyannote_output_dir"
    echo "  📊 切分資料集: $split_dir"
    echo ""
    
    # Step 1: UVR5 人聲分離狀態檢查和處理
    echo "🎵 Step 1: UVR5 人聲分離..."
    
    # 尋找原始音檔
    local episode_pattern
    if [ ${#episode_num} -eq 1 ]; then
        episode_pattern="第0${episode_num}集"
    elif [ ${#episode_num} -eq 2 ]; then
        episode_pattern="第${episode_num}集"
    else
        episode_pattern="第${episode_num}集"
    fi
    
    # 優先尋找 back_left.wav，其次是其他音檔
    local audio_file=$(find "$input_dir" -path "*${episode_pattern}*" -name "back_left.wav" | head -1)
    if [ -z "$audio_file" ]; then
        audio_file=$(find "$input_dir" -path "*${episode_pattern}*" \( -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" \) | head -1)
    fi
    
    if [ -z "$audio_file" ]; then
        echo "❌ 找不到第 $episode_num 集的音檔"
        return 1
    fi
    
    echo "  📥 原始音檔: $(basename "$audio_file")"
    
    # UVR5 輸出檔案 - 加上集數編號避免衝突
    local audio_basename=$(basename "$audio_file")
    local audio_name="${audio_basename%.*}"
    local audio_ext="${audio_basename##*.}"
    local uvr5_output_file="$uvr5_output_dir/${audio_name}_ep${episode_num}.${audio_ext}"
    
    # 為了讓 UVR5 正確處理，我們需要先複製檔案到 UVR5 輸出目錄
    local temp_input_file="$uvr5_output_dir/temp_${audio_name}_ep${episode_num}.${audio_ext}"
    
    # 檢查 UVR5 是否已完成（含檔案完整性檢查）
    if [ -f "$uvr5_output_file" ]; then
        # 檢查檔案大小是否合理（至少 1KB）
        local file_size=$(stat -c%s "$uvr5_output_file" 2>/dev/null || echo "0")
        if [ "$file_size" -gt 1024 ]; then
            echo "  ⏭️  UVR5 已完成: $(basename "$uvr5_output_file") (${file_size} bytes)"
        else
            echo "  ⚠️  UVR5 輸出檔案過小，重新處理: $(basename "$uvr5_output_file")"
            rm -f "$uvr5_output_file"
        fi
    fi
    
    if [ ! -f "$uvr5_output_file" ]; then
        echo "  📤 UVR5 輸出: $uvr5_output_file"
        
        # 檢查 UVR5 環境
        if ! check_uvr5_environment >/dev/null 2>&1; then
            echo "❌ UVR5 環境未準備就緒，請先檢查設定"
            return 1
        fi
        
        # 創建 UVR5 輸出目錄
        mkdir -p "$uvr5_output_dir"
        
        # 執行 UVR5 處理
        local python_cmd=$(detect_python)
        if [ -z "$python_cmd" ]; then
            echo "❌ 找不到 Python"
            return 1
        fi
        
        # 記錄 UVR5 開始時間
        local uvr5_start_time=$(date +%s)
        log_step_start "$episode_num" "UVR5人聲分離"
        
        # 先複製檔案到臨時位置，避免檔案名稱衝突
        cp "$audio_file" "$temp_input_file"
        
        echo "  🔄 處理中..."
        if $python_cmd "src/uvr5_cli.py" "$temp_input_file" --threads "${UVR5_MAX_WORKERS:-2}" --output-dir "$uvr5_output_dir" --no-backup; then
            # UVR5 會產生與輸入檔案同名的輸出檔案，我們需要重新命名
            local temp_output_file="$uvr5_output_dir/$(basename "$temp_input_file")"
            if [ -f "$temp_output_file" ]; then
                mv "$temp_output_file" "$uvr5_output_file"
                echo "  ✅ UVR5 人聲分離完成"
                log_step_end "$episode_num" "UVR5人聲分離" "$uvr5_start_time"
            else
                echo "❌ UVR5 輸出檔案不存在: $temp_output_file"
                log_step_failed "$episode_num" "UVR5人聲分離" "$uvr5_start_time"
                return 1
            fi
            
            # 清理臨時檔案
            rm -f "$temp_input_file"
        else
            echo "❌ UVR5 人聲分離失敗"
            log_step_failed "$episode_num" "UVR5人聲分離" "$uvr5_start_time"
            rm -f "$temp_input_file"
            return 1
        fi
    fi
    
    # Step 2: pyannote 處理狀態檢查和處理
    echo ""
    echo "🤖 Step 2: pyannote 處理..."
    
    # 檢查 pyannote 輸出是否已存在
    local episode_padded=$(printf "%03d" "$episode_num")
    local pyannote_episode_dir="$pyannote_output_dir"
    local pyannote_completed=false
    
    # 檢查：看是否有該集的輸出檔案（檢查任何語者的該集目錄）
    if [ -d "$pyannote_episode_dir" ] && [ -n "$(find "$pyannote_episode_dir" -mindepth 2 -maxdepth 2 -name "${episode_padded}" -type d | head -1)" ]; then
        echo "  ⏭️  pyannote 已完成: $pyannote_episode_dir"
        pyannote_completed=true
    else
        echo "  📥 UVR5 輸入: $uvr5_output_file"
        echo "  📤 pyannote 輸出: $pyannote_episode_dir"
        
        # 尋找字幕檔案
        local subtitle_pattern
        if [ ${#episode_num} -eq 1 ]; then
            subtitle_pattern="願望-00${episode_num}.txt"
        elif [ ${#episode_num} -eq 2 ]; then
            subtitle_pattern="願望-0${episode_num}.txt"
        else
            subtitle_pattern="願望-${episode_num}.txt"
        fi
        
        local subtitle_file=$(find "$input_dir" -name "$subtitle_pattern" | head -1)
        
        if [ -z "$subtitle_file" ]; then
            echo "❌ 找不到第 $episode_num 集的字幕檔案: $subtitle_pattern"
            return 1
        fi
        
        echo "  📝 字幕檔案: $(basename "$subtitle_file")"
        echo "  🔄 處理中..."
        
        # 記錄 pyannote 開始時間
        local pyannote_start_time=$(date +%s)
        log_step_start "$episode_num" "pyannote處理"
        
        # 執行 pyannote 處理
        if $python_cmd "src/pyannote_speaker_segmentation.py" \
            "$uvr5_output_file" "$subtitle_file" \
            --episode_num "$episode_num" \
            --output_dir "$pyannote_output_dir"; then
            echo "  ✅ pyannote 處理完成"
            log_step_end "$episode_num" "pyannote處理" "$pyannote_start_time"
        else
            echo "❌ pyannote 處理失敗"
            log_step_failed "$episode_num" "pyannote處理" "$pyannote_start_time"
            return 1
        fi
    fi
    
    # Step 3: 切分資料集狀態檢查和處理
    echo ""
    echo "📊 Step 3: 切分資料集..."
    
    # 檢查：看切分目錄是否有該集的目錄結構（任何語者的該集目錄）
    if [ -d "$split_dir/train" ] && [ -n "$(find "$split_dir/train" -mindepth 2 -maxdepth 2 -name "${episode_padded}" -type d | head -1)" ]; then
        echo "  ⏭️  切分已完成: $split_dir"
    else
        echo "  📥 輸入: $pyannote_output_dir"
        echo "  📤 輸出: $split_dir"
        echo "  🔄 處理中..."
        
        local test_ratio="${DEFAULT_TEST_RATIO:-0.2}"
        
        # 確保 python_cmd 變數正確設定
        local python_cmd=$(detect_python)
        if [ -z "$python_cmd" ]; then
            python_cmd="python3"
        fi
        
        # 記錄切分開始時間
        local split_start_time=$(date +%s)
        log_step_start "$episode_num" "切分資料集"
        
        if $python_cmd src/split_dataset.py \
            --processed_dir "$pyannote_output_dir" \
            --split_dir "$split_dir" \
            --method "episode" \
            --episode_num "$episode_num" \
            --test_ratio "$test_ratio"; then
            echo "  ✅ 切分完成"
            log_step_end "$episode_num" "切分資料集" "$split_start_time"
        else
            echo "❌ 切分失敗"
            log_step_failed "$episode_num" "切分資料集" "$split_start_time"
            return 1
        fi
    fi
    
    echo ""
    echo "🎉 第 $episode_num 集 UVR5 增強處理完成！"
    echo "📁 最終輸出：$split_dir (已進行人聲分離的訓練集)"
    
    return 0
}

# 批次處理多集 - UVR5 增強版本
smart_process_episodes_with_uvr5() {
    local episodes=("$@")
    
    if [ ${#episodes[@]} -eq 0 ]; then
        echo "❌ 請提供要處理的集數"
        echo "用法: smart_process_episodes_with_uvr5 1 2 3"
        return 1
    fi
    
    if [ ${#episodes[@]} -eq 1 ]; then
        echo "🚀 UVR5 增強智慧處理 - 第 ${episodes[0]} 集"
        echo "========================================"
    else
        echo "🚀 批次 UVR5 增強智慧處理 ${#episodes[@]} 集"
        echo "========================================"
    fi
    echo "💡 流程：UVR5 人聲分離 → pyannote 處理 → 切分資料集"
    echo ""
    
    local success_count=0
    local failed_episodes=()
    
    for episode in "${episodes[@]}"; do
        if smart_process_episode_with_uvr5 "$episode"; then
            ((success_count++))
        else
            failed_episodes+=("$episode")
        fi
        echo ""
    done
    
    echo "📊 批次 UVR5 處理結果："
    echo "✅ 成功: $success_count 集"
    if [ ${#failed_episodes[@]} -gt 0 ]; then
        echo "❌ 失敗: ${failed_episodes[*]}"
    fi
    
    return 0
}

# 批次智慧處理多集
smart_process_episodes() {
    local episodes=("$@")
    
    if [ ${#episodes[@]} -eq 0 ]; then
        echo "❌ 請提供要處理的集數"
        echo "用法: smart_process_episodes 1 2 3"
        return 1
    fi
    
    echo "🚀 批次智慧處理 ${#episodes[@]} 集"
    echo "================================"
    
    local success_count=0
    local failed_episodes=()
    
    for episode in "${episodes[@]}"; do
        if smart_process_episode "$episode"; then
            ((success_count++))
        else
            failed_episodes+=("$episode")
        fi
        echo ""
    done
    
    echo "📊 批次處理結果："
    echo "✅ 成功: $success_count 集"
    if [ ${#failed_episodes[@]} -gt 0 ]; then
        echo "❌ 失敗: ${failed_episodes[*]}"
    fi
    
    return 0
}

# 如果直接執行此腳本
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    if [ $# -eq 0 ]; then
        echo "智慧一條龍處理服務"
        echo "=================="
        echo "用法："
        echo "  $0 <集數>           # 處理單集"
        echo "  $0 1 2 3           # 處理多集"
        echo ""
        echo "範例："
        echo "  $0 1               # 處理第1集"
        echo "  $0 1 2 3 4 5       # 處理第1-5集"
        exit 1
    fi
    
    smart_process_episodes "$@"
fi
