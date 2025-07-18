#!/bin/bash

# Breeze ASR - Interactive Menu
# 互動式選單介面

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env file if exists
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# Function to show main menu
show_menu() {
    clear
    echo "🎤 Breeze ASR - Speaker Diarization Tool"
    echo "=========================================="
    echo ""
    echo "請選擇功能："
    echo "1. 處理集數 (Process Episodes)"
    echo "2. 處理並切分 (Process & Split)"
    echo "3. 切分訓練/測試集 (Split Dataset)"
    echo "4. 查看狀態 (View Status)"
    echo "5. 離開 (Exit)"
    echo ""
    echo -n "請輸入選項 [1-5]: "
}

# Function to process a single episode
process_single_episode() {
    local episode_num="$1"
    
    # Base directories
    BASE_DIR="../願望(音軌及字幕檔)"
    SUBTITLE_DIR="$BASE_DIR/願望(字幕檔)"
    OUTPUT_BASE_DIR="./output"
    
    # Find episode directory
    episode_padded=$(printf "%02d" "$episode_num")
    episode_dir=$(ls -1 "$BASE_DIR" 2>/dev/null | grep "第${episode_padded}集" | head -1)
    if [ -n "$episode_dir" ]; then
        episode_dir="$BASE_DIR/$episode_dir"
    fi
    
    if [ -z "$episode_dir" ]; then
        echo "❌ 找不到第 $episode_num 集的目錄"
        return 1
    fi
    
    # Find files
    subtitle_padded=$(printf "%03d" "$episode_num")
    subtitle_file="$SUBTITLE_DIR/願望-${subtitle_padded}.txt"
    audio_file="$episode_dir/back_left.wav"
    
    # Check if files exist
    if [ ! -f "$audio_file" ]; then
        echo "❌ 找不到音檔: $audio_file"
        return 1
    fi
    
    if [ ! -f "$subtitle_file" ]; then
        echo "❌ 找不到字幕檔: $subtitle_file"
        return 1
    fi
    
    # Use base output directory directly
    episode_output_dir="$OUTPUT_BASE_DIR"
    mkdir -p "$episode_output_dir"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    # Use python3 if available, otherwise python
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi
    
    # If in poetry environment, use poetry run
    if [ -f "pyproject.toml" ] && command -v poetry &> /dev/null; then
        if poetry env info &> /dev/null; then
            PYTHON_CMD="poetry run python"
        fi
    fi
    
    # Check if required packages are installed
    missing_packages=()
    for package in "pyannote.audio" "librosa" "soundfile" "numpy" "torch"; do
        if ! $PYTHON_CMD -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo "❌ 缺少套件: ${missing_packages[*]}"
        echo "請執行: pip install -r requirements.txt"
        return 1
    fi
    
    # Run the segmentation script
    $PYTHON_CMD src/pyannote_speaker_segmentation.py \
        "$audio_file" \
        "$subtitle_file" \
        --output_dir "$episode_output_dir" \
        --episode_num "$episode_num" \
        --min_duration 2.0 \
        --max_duration 15.0
    
    return $?
}

# Function to process episodes
process_episode() {
    echo ""
    echo "📺 處理集數"
    echo "=========="
    echo "1. 指定集數處理"
    echo "2. 處理所有集數"
    echo -n "請選擇 [1-2]: "
    read choice
    
    case "$choice" in
        1)
            echo ""
            echo "支援格式："
            echo "  單集: 1"
            echo "  多集: 1 3 5"
            echo "  範圍: 2-6"
            echo -n "請輸入集數: "
            read input
            process_custom_episodes "$input"
            ;;
        2)
            process_all_episodes
            ;;
        *)
            echo "❌ 無效選項"
            read -p "按 Enter 繼續..."
            ;;
    esac
}

# Function to process custom episodes
process_custom_episodes() {
    local input="$1"
    
    if [ -z "$input" ]; then
        echo "❌ 請輸入集數"
        read -p "按 Enter 繼續..."
        return
    fi
    
    # Parse input
    episodes=()
    
    if [[ "$input" =~ ^[0-9]+-[0-9]+$ ]]; then
        # Range format: 2-6
        start=$(echo "$input" | cut -d'-' -f1)
        end=$(echo "$input" | cut -d'-' -f2)
        
        if [ "$start" -le "$end" ]; then
            for ((i=start; i<=end; i++)); do
                episodes+=("$i")
            done
        else
            echo "❌ 範圍格式錯誤"
            read -p "按 Enter 繼續..."
            return
        fi
    else
        # Single or multiple episodes: 1 or 1 3 5
        for num in $input; do
            if [[ "$num" =~ ^[0-9]+$ ]]; then
                episodes+=("$num")
            else
                echo "❌ 無效的集數: $num"
                read -p "按 Enter 繼續..."
                return
            fi
        done
    fi
    
    if [ ${#episodes[@]} -eq 0 ]; then
        echo "❌ 沒有有效的集數"
        read -p "按 Enter 繼續..."
        return
    fi
    
    echo ""
    echo "🎵 準備處理 ${#episodes[@]} 集: ${episodes[*]}"
    echo ""
    
    # Process each episode
    success_count=0
    failed_episodes=()
    
    for episode in "${episodes[@]}"; do
        echo "🎵 處理第 $episode 集..."
        if process_single_episode "$episode"; then
            echo "✅ 第 $episode 集完成"
            ((success_count++))
        else
            echo "❌ 第 $episode 集失敗"
            failed_episodes+=("$episode")
        fi
        echo ""
    done
    
    echo "📊 處理結果："
    echo "✅ 成功: $success_count 集"
    
    if [ ${#failed_episodes[@]} -gt 0 ]; then
        echo "❌ 失敗: ${failed_episodes[*]}"
    fi
    
    echo ""
    read -p "按 Enter 繼續..."
}

# Function to process all episodes
process_all_episodes() {
    echo ""
    echo "📺 處理所有集數"
    echo "============="
    
    # Find all available episodes
    BASE_DIR="../願望(音軌及字幕檔)"
    available_episodes=()
    
    for episode_dir in "$BASE_DIR"/願望HD*第*集*; do
        if [ -d "$episode_dir" ]; then
            episode_num=$(basename "$episode_dir" | grep -o '[0-9]\+' | tail -1)
            if [ -n "$episode_num" ]; then
                available_episodes+=("$episode_num")
            fi
        fi
    done
    
    if [ ${#available_episodes[@]} -eq 0 ]; then
        echo "❌ 找不到任何集數"
        read -p "按 Enter 繼續..."
        return
    fi
    
    # Sort episodes
    IFS=$'\n' sorted_episodes=($(sort -n <<<"${available_episodes[*]}"))
    unset IFS
    
    echo "📋 找到 ${#sorted_episodes[@]} 集: ${sorted_episodes[*]}"
    echo ""
    echo -n "確定要處理所有集數嗎？[y/N]: "
    read confirm
    
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "取消處理"
        read -p "按 Enter 繼續..."
        return
    fi
    
    echo ""
    echo "🎵 開始處理所有集數..."
    echo ""
    
    # Process each episode
    success_count=0
    failed_episodes=()
    
    for episode in "${sorted_episodes[@]}"; do
        echo "🎵 處理第 $episode 集..."
        if process_single_episode "$episode"; then
            echo "✅ 第 $episode 集完成"
            ((success_count++))
        else
            echo "❌ 第 $episode 集失敗"
            failed_episodes+=("$episode")
        fi
        echo ""
    done
    
    echo "📊 處理結果："
    echo "✅ 成功: $success_count 集"
    
    if [ ${#failed_episodes[@]} -gt 0 ]; then
        echo "❌ 失敗: ${failed_episodes[*]}"
    fi
    
    echo ""
    read -p "按 Enter 繼續..."
}

# Function to process and split (一條龍服務)
process_and_split() {
    echo ""
    echo "🚀 處理並切分 (一條龍服務)"
    echo "======================"
    echo "1. 指定集數處理並切分"
    echo "2. 處理所有集數並切分"
    echo -n "請選擇 [1-2]: "
    read choice
    
    case "$choice" in
        1)
            echo ""
            echo "支援格式："
            echo "  單集: 1"
            echo "  多集: 1 3 5"
            echo "  範圍: 2-6"
            echo -n "請輸入集數: "
            read input
            
            echo ""
            echo "🎵 第一階段：處理集數..."
            if process_custom_episodes "$input"; then
                echo ""
                echo "🔄 第二階段：切分訓練/測試集..."
                perform_split
            else
                echo "❌ 處理失敗，取消切分"
            fi
            ;;
        2)
            echo ""
            echo "🎵 第一階段：處理所有集數..."
            if process_all_episodes; then
                echo ""
                echo "🔄 第二階段：切分訓練/測試集..."
                perform_split
            else
                echo "❌ 處理失敗，取消切分"
            fi
            ;;
        *)
            echo "❌ 無效選項"
            read -p "按 Enter 繼續..."
            ;;
    esac
}

# Function to perform split (internal use)
perform_split() {
    # Check if output directory exists
    if [ ! -d "output" ]; then
        echo "❌ 找不到 output 目錄"
        return 1
    fi
    
    # Count speakers (check both old and new format)
    speaker_count=$(find output -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    
    # Check for old format (episode_XXX directories)
    if [ "$speaker_count" -eq 0 ]; then
        episode_dirs=$(find output -maxdepth 1 -type d -name "episode_*" | wc -l)
        if [ "$episode_dirs" -gt 0 ]; then
            speaker_count=$(find output/episode_* -maxdepth 1 -type d -name "[0-9]*" | wc -l)
        fi
    fi
    
    if [ "$speaker_count" -eq 0 ]; then
        echo "❌ 找不到說話人資料"
        return 1
    fi
    
    echo "📊 找到 $speaker_count 個說話人"
    echo ""
    echo "按說話人切分 (80% 訓練集, 20% 測試集)"
    echo -n "確定要進行切分嗎？[y/N]: "
    read confirm
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔄 開始切分..."
        if python src/split_dataset.py output --method speaker --test_ratio 0.2; then
            echo "✅ 一條龍服務完成！"
            echo "📁 結果保存在 split_dataset/ 目錄"
            return 0
        else
            echo "❌ 切分失敗"
            return 1
        fi
    else
        echo "取消切分"
        return 1
    fi
}

# Function to split dataset
split_dataset() {
    echo ""
    echo "🔄 切分訓練/測試集"
    echo "================"
    
    # Check if output directory exists
    if [ ! -d "output" ]; then
        echo "❌ 找不到 output 目錄"
        echo "請先處理音檔再進行切分"
        read -p "按 Enter 繼續..."
        return
    fi
    
    # Count speakers (check both old and new format)
    speaker_count=$(find output -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    
    # Check for old format (episode_XXX directories)
    if [ "$speaker_count" -eq 0 ]; then
        episode_dirs=$(find output -maxdepth 1 -type d -name "episode_*" | wc -l)
        if [ "$episode_dirs" -gt 0 ]; then
            speaker_count=$(find output/episode_* -maxdepth 1 -type d -name "[0-9]*" | wc -l)
        fi
    fi
    
    if [ "$speaker_count" -eq 0 ]; then
        echo "❌ 找不到說話人資料"
        echo "請先處理音檔再進行切分"
        read -p "按 Enter 繼續..."
        return
    fi
    
    echo "📊 找到 $speaker_count 個說話人"
    echo ""
    echo "按說話人切分 (80% 訓練集, 20% 測試集)"
    echo -n "確定要進行切分嗎？[y/N]: "
    read confirm
    
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔄 開始切分..."
        if python src/split_dataset.py output --method speaker --test_ratio 0.2; then
            echo "✅ 切分完成！"
            echo "📁 結果保存在 split_dataset/ 目錄"
        else
            echo "❌ 切分失敗"
        fi
    else
        echo "取消切分"
    fi
    
    echo ""
    read -p "按 Enter 繼續..."
}

# Function to view status
view_status() {
    echo ""
    echo "📊 系統狀態"
    echo "=========="
    
    # Check output directory
    if [ -d "output" ]; then
        speaker_count=$(find output -maxdepth 1 -type d -name "[0-9]*" | wc -l)
        audio_count=$(find output -name "*.wav" | wc -l)
        echo "✅ 音檔處理: $speaker_count 個說話人, $audio_count 個音檔片段"
    else
        echo "❌ 尚未處理音檔"
    fi
    
    # Check split dataset
    if [ -d "split_dataset" ]; then
        train_count=$(find split_dataset/train -name "*.wav" 2>/dev/null | wc -l)
        test_count=$(find split_dataset/test -name "*.wav" 2>/dev/null | wc -l)
        echo "✅ 資料切分: $train_count 個訓練樣本, $test_count 個測試樣本"
    else
        echo "❌ 尚未切分資料集"
    fi
    
    # Check dependencies
    echo ""
    echo "📋 依賴檢查:"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        echo "✅ Python3"
    else
        echo "❌ Python3 未安裝"
    fi
    
    # Check HF Token
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "✅ Hugging Face Token"
    else
        echo "❌ 未設定 Hugging Face Token"
    fi
    
    echo ""
    read -p "按 Enter 繼續..."
}

# Main loop
while true; do
    show_menu
    read choice
    
    case "$choice" in
        1)
            process_episode
            ;;
        2)
            process_and_split
            ;;
        3)
            split_dataset
            ;;
        4)
            view_status
            ;;
        5)
            echo ""
            echo "👋 再見！"
            exit 0
            ;;
        *)
            echo ""
            echo "❌ 無效選項，請重新選擇"
            sleep 1
            ;;
    esac
done