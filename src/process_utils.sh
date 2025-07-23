#!/bin/bash

# Process utilities for SETVoicePrep interactive script
# Functions for episode processing and dataset splitting

# Function to check if input directory exists and has valid files
check_input_directory() {
    local input_dir="$1"
    
    if [ ! -d "$input_dir" ]; then
        echo "❌ 輸入目錄不存在: $input_dir"
        return 1
    fi
    
    # Check for audio files
    local audio_count=$(find "$input_dir" -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" | wc -l)
    if [ "$audio_count" -eq 0 ]; then
        echo "❌ 在目錄中找不到音訊檔案: $input_dir"
        return 1
    fi
    
    echo "✅ 找到 $audio_count 個音訊檔案"
    return 0
}

# Function to process a single episode
process_single_episode() {
    local episode_num="$1"
    local input_dir="$2"
    local output_dir="$3"
    
    echo ""
    echo "🎵 處理集數 $episode_num"
    echo "=================="
    
    # Validate inputs
    if [ -z "$episode_num" ] || [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
        echo "❌ 缺少必要參數"
        echo "用法: process_single_episode <集數> <輸入目錄> <輸出目錄>"
        return 1
    fi
    
    # Check if input directory exists
    if ! check_input_directory "$input_dir"; then
        return 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Find audio files for the specified episode
    local episode_pattern
    if [ ${#episode_num} -eq 1 ]; then
        episode_pattern="第0${episode_num}集"
    elif [ ${#episode_num} -eq 2 ]; then
        episode_pattern="第${episode_num}集"
    else
        episode_pattern="第${episode_num}集"
    fi
    
    local audio_files=$(find "$input_dir" -path "*${episode_pattern}*" \( -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" -o -name "*.flac" \))
    
    if [ -z "$audio_files" ]; then
        echo "❌ 找不到集數 $episode_num 的音訊檔案"
        return 1
    fi
    
    
    # Run pyannote speaker segmentation
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    echo ""
    echo "🔄 開始處理..."
    
    # Find subtitle file for this episode
    local subtitle_pattern
    if [ ${#episode_num} -eq 1 ]; then
        subtitle_pattern="願望-00${episode_num}.txt"
    elif [ ${#episode_num} -eq 2 ]; then
        subtitle_pattern="願望-0${episode_num}.txt"
    else
        subtitle_pattern="願望-${episode_num}.txt"
    fi
    
    local subtitle_file=$(find "$input_dir" -name "$subtitle_pattern")
    
    if [ -z "$subtitle_file" ]; then
        echo "❌ 找不到集數 $episode_num 的字幕檔案: $subtitle_pattern"
        return 1
    fi
    
    echo "📝 找到字幕檔案: $(basename "$subtitle_file")"
    
    # Process each audio file individually - use a simple approach
    local episode_dir=$(find "$input_dir" -type d -path "*${episode_pattern}*" | head -1)
    
    if [ -z "$episode_dir" ]; then
        echo "❌ 找不到集數 $episode_num 的目錄"
        return 1
    fi
    
    echo "📁 處理目錄: $(basename "$episode_dir")"
    
    # Process back_left.wav only
    local audio_file="$episode_dir/back_left.wav"
    
    if [ -f "$audio_file" ]; then
        echo "處理: $(basename "$audio_file")"
        
        # Add embedding verification options
        local verification_args=""
        if [ -n "${SIMILARITY_THRESHOLD:-}" ]; then
            verification_args="$verification_args --similarity_threshold $SIMILARITY_THRESHOLD"
        fi
        
        if [ -n "${VOICE_ACTIVITY_THRESHOLD:-}" ]; then
            verification_args="$verification_args --voice_activity_threshold $VOICE_ACTIVITY_THRESHOLD"
        fi
        
        # Hybrid segmentation mode is default and only supported mode
        echo "🎭 使用混合分段模式 (Diarization + 字幕)"
        
        PYTHONIOENCODING=UTF-8 $python_cmd src/pyannote_speaker_segmentation.py \
            "$audio_file" \
            "$subtitle_file" \
            --episode_num "$episode_num" \
            --output_dir "$output_dir" \
            --force \
            $verification_args > debug_log.txt 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ 完成: $(basename "$audio_file")"
        else
            echo "❌ 失敗: $(basename "$audio_file")"
        fi
    else
        echo "❌ 找不到 back_left.wav"
        return 1
    fi
    
    echo ""
    echo "✅ 集數 $episode_num 處理完成"
    return 0
}

# Function to process multiple episodes
process_multiple_episodes() {
    local episodes_list="$1"
    local input_dir="$2"
    local output_dir="$3"
    
    echo ""
    echo "🎵 處理多個集數"
    echo "==============="
    
    if [ -z "$episodes_list" ] || [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
        echo "❌ 缺少必要參數"
        return 1
    fi
    
    # Parse episodes list
    local episodes
    if ! episodes=$(validate_episode_input "$episodes_list"); then
        return 1
    fi
    
    local total_episodes=$(echo "$episodes" | wc -l)
    echo "📊 將處理 $total_episodes 個集數: $(echo "$episodes" | tr '\n' ' ')"
    
    echo "🚀 開始處理..."
    
    local success_count=0
    local fail_count=0
    
    echo "$episodes" | while read -r episode; do
        if [ -n "$episode" ]; then
            if process_single_episode "$episode" "$input_dir" "$output_dir"; then
                ((success_count++))
            else
                ((fail_count++))
            fi
            echo ""
        fi
    done
    
    echo "📊 處理完成統計:"
    echo "  ✅ 成功: $success_count"
    echo "  ❌ 失敗: $fail_count"
    
    return 0
}

# Function to process all episodes in a directory
process_all_episodes() {
    local input_dir="$1"
    local output_dir="$2"
    
    echo ""
    echo "🎵 處理所有集數"
    echo "==============="
    
    if ! check_input_directory "$input_dir"; then
        return 1
    fi
    
    # Find all unique episode numbers from directory names
    local episodes=$(find "$input_dir" -type d -name "*第*集*" | \
        sed 's/.*第0*\([0-9]\+\)集.*/\1/' | \
        sort -u -n)
    
    if [ -z "$episodes" ]; then
        echo "❌ 無法識別任何集數"
        return 1
    fi
    
    local total_episodes=$(echo "$episodes" | wc -l)
    echo "📊 找到 $total_episodes 個集數: $(echo "$episodes" | tr '\n' ' ')"
    
    echo "🚀 開始處理..."
    
    local success_count=0
    local fail_count=0
    
    echo "$episodes" | while read -r episode; do
        if [ -n "$episode" ]; then
            if process_single_episode "$episode" "$input_dir" "$output_dir"; then
                ((success_count++))
            else
                ((fail_count++))
            fi
            echo ""
        fi
    done
    
    echo "📊 處理完成統計:"
    echo "  ✅ 成功: $success_count"
    echo "  ❌ 失敗: $fail_count"
    
    return 0
}

# Function to split dataset into train/test
split_dataset() {
    local processed_dir="$1"
    local output_dir="$2"
    local test_ratio="${3:-0.2}"
    
    echo ""
    echo "📊 切分資料集"
    echo "============="
    
    if [ ! -d "$processed_dir" ]; then
        echo "❌ 處理後目錄不存在: $processed_dir"
        return 1
    fi
    
    # Check for processed files
    local processed_count=$(find "$processed_dir" -name "*.wav" | wc -l)
    if [ "$processed_count" -eq 0 ]; then
        echo "❌ 在處理後目錄中找不到音訊檔案: $processed_dir"
        return 1
    fi
    
    echo "📁 找到 $processed_count 個處理後的音訊檔案"
    echo "📋 測試集比例: $test_ratio"
    
    echo "🚀 開始切分..."
    
    # Run dataset splitting script
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    echo ""
    echo "🔄 開始切分..."
    
    $python_cmd "src/split_dataset.py" \
        "$processed_dir" \
        --output_dir "$output_dir" \
        --test_ratio "$test_ratio"
    
    if [ $? -eq 0 ]; then
        echo "✅ 資料集切分完成"
        
        # Show statistics
        local train_count=$(find "$output_dir/train" -name "*.wav" 2>/dev/null | wc -l)
        local test_count=$(find "$output_dir/test" -name "*.wav" 2>/dev/null | wc -l)
        
        echo ""
        echo "📊 切分統計:"
        echo "  🎓 訓練集: $train_count 檔案"
        echo "  🧪 測試集: $test_count 檔案"
        
        return 0
    else
        echo "❌ 資料集切分失敗"
        return 1
    fi
}

# Function to process and split in one step
process_and_split() {
    local episodes_input="$1"
    local input_dir="$2"
    local processed_dir="$3"
    local split_dir="$4"
    local test_ratio="${5:-0.2}"
    
    echo ""
    echo "🚀 處理並切分"
    echo "============="
    echo "📋 集數: $episodes_input"
    echo "📁 輸入目錄: $input_dir"
    echo "📁 處理目錄: $processed_dir"
    echo "📁 切分目錄: $split_dir"
    echo "📋 測試集比例: $test_ratio"
    echo ""
    
    # Always confirm for the main "process and split" function to avoid accidents
    if ! get_confirmation "確定要開始處理並切分嗎？"; then
        echo "❌ 已取消"
        return 1
    fi
    
    # Step 1: Process episodes
    echo ""
    echo "📝 步驟 1/2: 處理集數"
    echo "==================="
    
    if [ "$episodes_input" = "all" ]; then
        if ! process_all_episodes "$input_dir" "$processed_dir"; then
            echo "❌ 集數處理失敗"
            return 1
        fi
    else
        if ! process_multiple_episodes "$episodes_input" "$input_dir" "$processed_dir"; then
            echo "❌ 集數處理失敗"
            return 1
        fi
    fi
    
    # Step 2: Split dataset (incremental)
    echo ""
    echo "📝 步驟 2/2: 切分資料集 (增量)"
    echo "========================="
    
    # Simple approach: just run split on all data
    echo "🔄 更新切分資料集..."
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    $python_cmd "src/split_dataset.py" \
        "$processed_dir" \
        --output_dir "$split_dir" \
        --test_ratio "$test_ratio"
    
    if [ $? -eq 0 ]; then
        echo "✅ 資料集切分完成"
        
        # Show statistics
        local train_count=$(find "$split_dir/train" -name "*.wav" 2>/dev/null | wc -l)
        local test_count=$(find "$split_dir/test" -name "*.wav" 2>/dev/null | wc -l)
        
        echo ""
        echo "📊 切分統計:"
        echo "  🎓 訓練集: $train_count 檔案"
        echo "  🧪 測試集: $test_count 檔案"
    else
        echo "❌ 資料集切分失敗"
        return 1
    fi
    
    echo ""
    echo "🎉 處理並切分完成！"
    echo "==================="
    echo "📁 處理後檔案: $processed_dir"
    echo "📁 切分後檔案: $split_dir"
    
    return 0
}

# Menu function for processing episodes
process_episodes_menu() {
    while true; do
        echo ""
        echo "🎵 處理集數"
        echo "=========="
        echo "1. 處理指定集數"
        echo "2. 處理所有集數"
        echo "3. 返回主選單"
        echo ""
        echo -n "請選擇 [1-3]: "
        read choice
        
        case "$choice" in
            1)
                echo ""
                echo "🎯 處理指定集數"
                echo "==============="
                echo "提示: 可輸入單一集數 (如: 5) 或範圍 (如: 2-6) 或多個集數 (如: 1 3 5)"
                echo -n "請輸入集數: "
                read episodes_input
                
                echo -n "輸入目錄 [預設: ${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}]: "
                read input_dir
                input_dir="${input_dir:-${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}}"
                
                echo -n "輸出目錄 [預設: ${DEFAULT_PROCESSED_DIR:-output}]: "
                read output_dir
                output_dir="${output_dir:-${DEFAULT_PROCESSED_DIR:-output}}"
                
                process_multiple_episodes "$episodes_input" "$input_dir" "$output_dir"
                pause_for_input
                ;;
            2)
                echo ""
                echo "🌟 處理所有集數"
                echo "==============="
                echo -n "輸入目錄 [預設: ${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}]: "
                read input_dir
                input_dir="${input_dir:-${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}}"
                
                echo -n "輸出目錄 [預設: ${DEFAULT_PROCESSED_DIR:-output}]: "
                read output_dir
                output_dir="${output_dir:-${DEFAULT_PROCESSED_DIR:-output}}"
                
                process_all_episodes "$input_dir" "$output_dir"
                pause_for_input
                ;;
            3)
                return
                ;;
            *)
                echo "❌ 無效選項"
                pause_for_input
                ;;
        esac
    done
}

# Menu function for process and split
process_and_split_menu() {
    echo ""
    echo "🚀 處理並切分"
    echo "============"
    echo "提示: 此功能將先處理集數，然後自動切分為訓練/測試集"
    echo ""
    
    echo "集數選項:"
    echo "  - 單一集數: 5"
    echo "  - 範圍: 2-6"
    echo "  - 多個集數: 1 3 5"
    echo "  - 所有集數: all"
    echo ""
    echo -n "請輸入集數: "
    read episodes_input
    
    echo -n "輸入目錄 [預設: ${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}]: "
    read input_dir
    input_dir="${input_dir:-${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}}"
    
    echo -n "處理目錄 [預設: ${DEFAULT_PROCESSED_DIR:-output}]: "
    read processed_dir
    processed_dir="${processed_dir:-${DEFAULT_PROCESSED_DIR:-output}}"
    
    echo -n "切分目錄 [預設: ${DEFAULT_SPLIT_DIR:-split_dataset}]: "
    read split_dir
    split_dir="${split_dir:-${DEFAULT_SPLIT_DIR:-split_dataset}}"
    
    echo -n "測試集比例 [預設: ${DEFAULT_TEST_RATIO:-0.2}]: "
    read test_ratio
    test_ratio="${test_ratio:-${DEFAULT_TEST_RATIO:-0.2}}"
    
    process_and_split "$episodes_input" "$input_dir" "$processed_dir" "$split_dir" "$test_ratio"
    pause_for_input
}

# Menu function for split dataset only
split_dataset_menu() {
    echo ""
    echo "📊 切分訓練/測試集"
    echo "================"
    
    echo -n "處理後目錄 [預設: ${DEFAULT_PROCESSED_DIR:-output}]: "
    read processed_dir
    processed_dir="${processed_dir:-${DEFAULT_PROCESSED_DIR:-output}}"
    
    echo -n "輸出目錄 [預設: ${DEFAULT_SPLIT_DIR:-split_dataset}]: "
    read output_dir
    output_dir="${output_dir:-${DEFAULT_SPLIT_DIR:-split_dataset}}"
    
    echo -n "測試集比例 [預設: ${DEFAULT_TEST_RATIO:-0.2}]: "
    read test_ratio
    test_ratio="${test_ratio:-${DEFAULT_TEST_RATIO:-0.2}}"
    
    split_dataset "$processed_dir" "$output_dir" "$test_ratio"
    pause_for_input
}