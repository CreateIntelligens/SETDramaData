#!/bin/bash

# Status utilities for etl interactive script
# Functions for viewing processing status and system information

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_utils.sh"

# Function to show system status (simplified)
show_system_status() {
    # No longer shows title - handled by main show_status function
    return 0
}

# Function to show processing state
show_processing_state() {
    echo "📊 處理狀態"
    echo "=========="
    
    if [ ! -f "processing_state.json" ]; then
        echo "❌ 處理狀態檔案不存在"
        echo "   還沒有處理過任何集數"
        return
    fi
    
    # Create temporary Python script to read status
    cat > show_status_temp.py << 'EOF'
import json
try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    processed = state.get('processed_episodes', [])
    last_id = state.get('last_used_speaker_id', -1)
    
    print(f'📺 已處理集數: {len(processed)} 集')
    if processed:
        sorted_episodes = sorted(processed)
        print(f'   集數列表: {", ".join(map(str, sorted_episodes))}')
    
    print(f'🎭 最後使用的 Speaker ID: {last_id}')
    
except Exception as e:
    print(f'❌ 讀取狀態檔案失敗: {e}')
EOF
    
    run_python_script show_status_temp.py
    echo ""
}

# Function to show output directory status
show_output_status() {
    echo "📁 輸出目錄狀態"
    echo "=============="
    
    if [ ! -d "output" ]; then
        echo "❌ output 目錄不存在"
        return
    fi
    
    # Count files and directories
    local speaker_dirs=$(find output -maxdepth 1 -type d | wc -l)
    speaker_dirs=$((speaker_dirs - 1))  # Subtract output directory itself
    
    local wav_files=$(find output -name "*.wav" | wc -l)
    local txt_files=$(find output -name "*.txt" | wc -l)
    local tsv_files=$(find output -name "*.tsv" | wc -l)
    
    echo "🎵 音頻檔案: $wav_files 個"
    echo "📝 文字檔案: $txt_files 個"
    
    # Show episodes summary
    echo ""
    echo "📺 已處理集數檔案統計:"
    
    # Get processed episodes from state
    if [ -f "processing_state.json" ]; then
        # Create temporary Python script to get episodes
        cat > episode_list_temp.py << 'EOF'
import json
try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    processed = sorted(state.get('processed_episodes', []))
    for ep in processed:
        print(ep)
except:
    pass
EOF
        
        local python_cmd=$(detect_python)
        if [ -n "$python_cmd" ]; then
            $python_cmd episode_list_temp.py | head -10 | while read -r episode; do
                if [ -n "$episode" ] && [[ "$episode" =~ ^[0-9]+$ ]]; then
                    local episode_padded=$(printf "%03d" "$episode" 2>/dev/null)
                    if [ $? -eq 0 ]; then
                        local total_files=$(find output -path "*/$episode_padded/*" -type f 2>/dev/null | wc -l)
                        if [ "$total_files" -gt 0 ]; then
                            echo "   第 $episode 集: $total_files 個檔案"
                        fi
                    fi
                fi
            done
            rm -f episode_list_temp.py
        fi
    fi
    
    echo ""
}

# Function to show split dataset status  
show_split_dataset_status() {
    echo "📊 切分資料集狀態"
    echo "================"
    
    if [ ! -d "split_dataset" ]; then
        echo "❌ split_dataset 目錄不存在"
        return
    fi
    
    # Check train and test directories
    local train_speakers=0
    local test_speakers=0
    local train_files=0
    local test_files=0
    
    if [ -d "split_dataset/train" ]; then
        train_speakers=$(find split_dataset/train -maxdepth 1 -type d | wc -l)
        train_speakers=$((train_speakers - 1))  # Subtract train directory itself
        train_files=$(find split_dataset/train -name "*.wav" | wc -l)
    fi
    
    if [ -d "split_dataset/test" ]; then
        test_speakers=$(find split_dataset/test -maxdepth 1 -type d | wc -l)
        test_speakers=$((test_speakers - 1))  # Subtract test directory itself
        test_files=$(find split_dataset/test -name "*.wav" | wc -l)
    fi
    
    echo "🚂 訓練集: $train_files 個音頻檔案"
    echo "🧪 測試集: $test_files 個音頻檔案"
    echo "📊 總計: $((train_files + test_files)) 個檔案"
    
    # Show actual episodes in split dataset by reading from processing state
    if [ -f "processing_state.json" ]; then
        local python_cmd=$(detect_python)
        if [ -n "$python_cmd" ]; then
            local episodes_info=$($python_cmd -c "
import json
try:
    with open('processing_state.json', 'r') as f:
        state = json.load(f)
    processed = sorted(state.get('processed_episodes', []))
    if processed:
        print(f'{len(processed)} 集 ({', '.join(map(str, processed))})')
    else:
        print('0 集')
except:
    print('無法讀取')
" 2>/dev/null)
            
            if [ -n "$episodes_info" ]; then
                echo "📺 涵蓋集數: $episodes_info"
            fi
        fi
    fi
    
    echo ""
}

# Function to show episode-specific information
show_episode_info() {
    echo -n "請輸入要查看的集數: "
    read episode_input
    
    if [ -z "$episode_input" ] || ! [[ "$episode_input" =~ ^[0-9]+$ ]]; then
        echo "❌ 請輸入有效的集數"
        return
    fi
    
    local episode_padded=$(printf "%03d" "$episode_input")
    
    echo ""
    echo "📺 第 $episode_input 集詳細資訊"
    echo "==================="
    
    # Check if episode exists in processing state
    if [ -f "processing_state.json" ]; then
        # Create temporary Python script
        cat > check_episode_temp.py << EOF
import json
try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    processed = state.get('processed_episodes', [])
    ranges = state.get('episode_speaker_ranges', {})
    
    episode_num = $episode_input
    episode_str = str(episode_num)
    
    if episode_num in processed:
        print('✅ 已處理')
        
        if episode_str in ranges:
            data = ranges[episode_str]
            start_id = data.get('start', 'N/A')
            end_id = data.get('end', 'N/A')
            mapping = data.get('mapping', {})
            print(f'🎭 Speaker ID 範圍: {start_id}-{end_id}')
            print(f'📊 Speaker 數量: {len(mapping)}')
        else:
            print('⚠️  範圍資訊缺失')
    else:
        print('❌ 未處理')
        
except Exception as e:
    print(f'❌ 讀取狀態失敗: {e}')
EOF
        
        run_python_script check_episode_temp.py
    fi
    
    # Check output directory
    echo ""
    echo "📁 輸出檔案:"
    local found_dirs=0
    local total_files=0
    
    for speaker_dir in output/*/; do
        if [ -d "$speaker_dir" ]; then
            local episode_dir="$speaker_dir$episode_padded"
            if [ -d "$episode_dir" ]; then
                local speaker=$(basename "$speaker_dir")
                local file_count=$(find "$episode_dir" -type f | wc -l)
                echo "   Speaker $speaker: $file_count 個檔案"
                ((found_dirs++))
                ((total_files += file_count))
            fi
        fi
    done
    
    if [ $found_dirs -eq 0 ]; then
        echo "   ❌ 無輸出檔案"
    else
        echo "   📊 總計: $found_dirs 個 speaker 目錄，$total_files 個檔案"
    fi
    
    # Check split dataset
    echo ""
    echo "📊 切分資料集:"
    if [ -d "split_dataset" ]; then
        local train_files=$(find split_dataset/train -path "*/$episode_padded/*" -type f 2>/dev/null | wc -l)
        local test_files=$(find split_dataset/test -path "*/$episode_padded/*" -type f 2>/dev/null | wc -l)
        
        echo "   🚂 訓練集: $train_files 個檔案"
        echo "   🧪 測試集: $test_files 個檔案"
    else
        echo "   ❌ 切分資料集不存在"
    fi
    
    echo ""
}

# Main status display function
show_status() {
    while true; do
        echo ""
        echo "📊 查看狀態"
        echo "=========="
        show_system_status
        show_processing_state
        show_output_status
        show_split_dataset_status
        
        echo "詳細選項:"
        echo "1. 查看特定集數資訊"
        echo "2. 刷新狀態"
        echo "3. 返回主選單"
        echo ""
        echo -n "請選擇 [1-3]: "
        read choice
        
        case "$choice" in
            1)
                show_episode_info
                pause_for_input
                ;;
            2)
                continue  # Just refresh by continuing the loop
                ;;
            3)
                break
                ;;
            *)
                echo "❌ 無效選項，請重新選擇"
                pause_for_input
                ;;
        esac
    done
}