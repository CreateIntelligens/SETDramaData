#!/bin/bash

# Cleanup utilities for Breeze ASR interactive script
# All data cleaning and removal functions

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_utils.sh"

# Function to clean all output files
clean_output_files() {
    echo ""
    echo "🗑️ 清除所有輸出檔案"
    echo "=================="
    
    if ! check_directory "output"; then
        pause_for_input
        return
    fi
    
    # Count files to be deleted
    wav_count=$(find output -name "*.wav" 2>/dev/null | wc -l)
    txt_count=$(find output -name "*.txt" 2>/dev/null | wc -l)
    tsv_count=$(find output -name "*.tsv" 2>/dev/null | wc -l)
    
    echo "📊 將刪除："
    echo "  音頻檔案: $wav_count 個"
    echo "  文字檔案: $txt_count 個"
    echo "  標註檔案: $tsv_count 個"
    echo ""
    echo "⚠️  這將刪除所有處理後的音頻和轉錄檔案，但不會影響處理狀態記錄！"
    echo "💡 如需重置處理狀態，請另外選擇「重置處理狀態」"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        echo "🗑️ 刪除中..."
        if rm -rf output; then
            echo "✅ 所有輸出檔案已清除（處理狀態保留）"
        else
            echo "❌ 清除失敗"
        fi
    else
        echo "取消操作"
    fi
    
    echo ""
    pause_for_input
}

# Function to reset processing state
reset_processing_state() {
    echo ""
    echo "🔄 重置處理狀態"
    echo "=============="
    
    if [ ! -f "processing_state.json" ]; then
        echo "❌ 處理狀態檔案不存在"
        pause_for_input
        return
    fi
    
    echo "📋 當前處理狀態："
    
    # Create temporary Python script to show detailed status
    cat > reset_status_temp.py << 'EOF'
import json
try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    processed = state.get('processed_episodes', [])
    last_id = state.get('last_used_speaker_id', -1)
    ranges = state.get('episode_speaker_ranges', {})
    
    print(f'已處理集數: {sorted(processed)}')
    print(f'總集數: {len(processed)}')
    print(f'最後使用的 Speaker ID: {last_id}')
    
    if ranges:
        print('\n集數詳情:')
        for ep in sorted(processed):
            ep_str = str(ep)
            if ep_str in ranges:
                range_info = ranges[ep_str]
                start = range_info.get('start', 'N/A')
                end = range_info.get('end', 'N/A')
                mapping = range_info.get('mapping', {})
                speakers = len(mapping)
                print(f'  集數 {ep}: Speaker ID {start}-{end} ({speakers} 個說話者)')
            else:
                print(f'  集數 {ep}: 無詳細資訊')
    
except Exception as e:
    print(f'無法讀取狀態檔案: {e}')
EOF
    
    run_python_script reset_status_temp.py
    
    echo ""
    echo "⚠️  這將重置所有處理狀態記錄，但不會刪除已處理的檔案！"
    echo "💡 如需同時刪除檔案，請選擇「全部清除」"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        echo "🔄 重置中..."
        if rm -f "processing_state.json"; then
            echo "✅ 處理狀態已重置（檔案保留）"
        else
            echo "❌ 重置失敗"
        fi
    else
        echo "取消操作"
    fi
    
    echo ""
    pause_for_input
}

# Function to clean split dataset
clean_split_dataset() {
    echo ""
    echo "🗑️ 清除切分資料集"
    echo "================"
    
    if ! check_directory "split_dataset"; then
        pause_for_input
        return
    fi
    
    # Count files to be deleted with detailed breakdown
    train_count=$(find split_dataset/train -name "*.wav" 2>/dev/null | wc -l)
    test_count=$(find split_dataset/test -name "*.wav" 2>/dev/null | wc -l)
    train_txt_count=$(find split_dataset/train -name "*.txt" 2>/dev/null | wc -l)
    test_txt_count=$(find split_dataset/test -name "*.txt" 2>/dev/null | wc -l)
    
    # Count episodes in split dataset
    train_episodes=$(find split_dataset/train -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | wc -l)
    test_episodes=$(find split_dataset/test -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | wc -l)
    
    echo "📊 切分資料集狀態："
    echo "  訓練集: $train_count 音頻檔 + $train_txt_count 文字檔 ($train_episodes 集)"
    echo "  測試集: $test_count 音頻檔 + $test_txt_count 文字檔 ($test_episodes 集)"
    echo "  總計: $((train_count + test_count)) 音頻檔"
    echo ""
    echo "⚠️  這只會刪除切分資料集，不影響原始處理檔案 (output/) 和狀態記錄！"
    echo "💡 原始處理檔案仍可重新切分"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        echo "🗑️ 刪除中..."
        if rm -rf split_dataset; then
            echo "✅ 切分資料集已清除（原始檔案保留）"
        else
            echo "❌ 清除失敗"
        fi
    else
        echo "取消操作"
    fi
    
    echo ""
    pause_for_input
}

# Function to clean specific episodes
clean_specific_episodes() {
    echo ""
    echo "🗑️ 清除特定集數處理結果"
    echo "===================="
    
    if ! check_directory "output"; then
        pause_for_input
        return
    fi
    
    # Show available episodes with detailed status
    if [ -f "processing_state.json" ]; then
        echo "📋 當前狀態："
        
        # Create temporary Python script
        cat > show_episodes_temp.py << 'EOF'
import json
import os

try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    processed = state.get('processed_episodes', [])
    ranges = state.get('episode_speaker_ranges', {})
    
    print(f'已處理集數: {sorted(processed)}')
    print(f'總計: {len(processed)} 集')
    
    # Check which episodes have output files
    if os.path.exists('output'):
        episodes_with_files = set()
        for root, dirs, files in os.walk('output'):
            if files:
                # Extract episode number from path
                parts = root.split(os.sep)
                for part in parts:
                    if part.isdigit() and len(part) == 3:
                        episodes_with_files.add(int(part.lstrip('0') or '0'))
        
        if episodes_with_files:
            missing_files = set(processed) - episodes_with_files
            if missing_files:
                print(f'狀態記錄有但檔案缺失: {sorted(missing_files)}')
        
        print(f'output/ 中有檔案的集數: {sorted(episodes_with_files)}')
    
    # Check split dataset
    if os.path.exists('split_dataset'):
        split_episodes = set()
        for subset in ['train', 'test']:
            subset_path = f'split_dataset/{subset}'
            if os.path.exists(subset_path):
                for root, dirs, files in os.walk(subset_path):
                    if files:
                        parts = root.split(os.sep)
                        for part in parts:
                            if part.isdigit() and len(part) == 3:
                                split_episodes.add(int(part.lstrip('0') or '0'))
        
        if split_episodes:
            print(f'split_dataset/ 中有檔案的集數: {sorted(split_episodes)}')
        else:
            print('split_dataset/ 中無檔案')
    else:
        print('split_dataset/ 不存在')
        
except Exception as e:
    print(f'無法讀取狀態: {e}')
EOF
        
        run_python_script show_episodes_temp.py
    else
        echo "📋 無處理狀態記錄"
        
        # Still check for files
        if [ -d "output" ]; then
            echo "📁 output/ 目錄存在，檢查檔案..."
            episodes_in_output=$(find output -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | sed 's/.*\///;s/^0*//' | sort -n | uniq)
            if [ -n "$episodes_in_output" ]; then
                echo "  有檔案的集數: $(echo $episodes_in_output | tr '\n' ' ')"
            else
                echo "  無檔案"
            fi
        fi
    fi
    
    echo ""
    echo "支援格式："
    echo "  單集: 1"
    echo "  多集: 1 3 5"
    echo "  範圍: 2-6"
    echo -n "請輸入要清除的集數: "
    read input
    
    # Validate input and get episodes array
    episodes_output=$(validate_episode_input "$input")
    if [ $? -ne 0 ]; then
        echo "$episodes_output"
        pause_for_input
        return
    fi
    
    # Convert output to array
    readarray -t episodes <<< "$episodes_output"
    
    echo ""
    echo "📊 準備清除 ${#episodes[@]} 集: ${episodes[*]}"
    echo ""
    echo "⚠️  這將刪除指定集數的所有處理結果！"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        echo "🗑️ 清除中..."
        
        success_count=0
        failed_episodes=()
        
        for episode in "${episodes[@]}"; do
            episode_padded=$(printf "%03d" "$episode")
            
            echo "🔍 搜尋第 $episode 集的檔案..."
            
            # Find all speaker directories containing this episode (format: SPEAKER_ID/EPISODE/)
            found_dirs=()
            files_count=0
            
            # Search all speaker directories for this episode
            for speaker_dir in output/*/; do
                if [ -d "$speaker_dir" ]; then
                    episode_dir="$speaker_dir$episode_padded"
                    if [ -d "$episode_dir" ]; then
                        found_dirs+=("$episode_dir")
                        dir_files=$(find "$episode_dir" -type f \( -name "*.wav" -o -name "*.txt" -o -name "*.tsv" \) | wc -l)
                        files_count=$((files_count + dir_files))
                    fi
                fi
            done
            
            if [ ${#found_dirs[@]} -gt 0 ]; then
                echo "📊 找到 ${#found_dirs[@]} 個目錄，$files_count 個檔案"
                
                # Remove all found episode directories
                all_success=true
                deleted_count=0
                for episode_dir in "${found_dirs[@]}"; do
                    if rm -rf "$episode_dir"; then
                        ((deleted_count++))
                        
                        # Remove parent speaker directory if empty
                        speaker_parent=$(dirname "$episode_dir")
                        cleanup_empty_directory "$speaker_parent" >/dev/null 2>&1
                    else
                        all_success=false
                    fi
                done
                
                echo "✅ 已刪除 output 中的 $deleted_count 個目錄"
                
                # Also remove from split_dataset if exists
                if [ -d "split_dataset" ]; then
                    # Count and remove files for this episode across all speakers
                    train_files=$(find split_dataset/train -path "*/$episode_padded/*" -type f 2>/dev/null | wc -l)
                    test_files=$(find split_dataset/test -path "*/$episode_padded/*" -type f 2>/dev/null | wc -l)
                    
                    if [ "$train_files" -gt 0 ] || [ "$test_files" -gt 0 ]; then
                        echo "✅ 已清除切分資料集中的 $((train_files + test_files)) 個檔案"
                        
                        # Remove episode directories from ALL speakers
                        find split_dataset/train -path "*/$episode_padded" -type d -exec rm -rf {} + 2>/dev/null
                        find split_dataset/test -path "*/$episode_padded" -type d -exec rm -rf {} + 2>/dev/null
                        
                        # Clean up empty speaker directories
                        find split_dataset/train -maxdepth 1 -type d -empty -exec rmdir {} + 2>/dev/null
                        find split_dataset/test -maxdepth 1 -type d -empty -exec rmdir {} + 2>/dev/null
                    fi
                fi
                
                if [ "$all_success" = true ]; then
                    ((success_count++))
                else
                    failed_episodes+=("$episode")
                fi
            else
                echo "⚠️  找不到第 $episode 集的檔案"
                failed_episodes+=("$episode")
            fi
            
            echo ""
        done
        
        # Update processing state - only for successfully deleted episodes
        if [ -f "processing_state.json" ] && [ "$success_count" -gt 0 ]; then
            
            # Create list of successfully deleted episodes
            successfully_deleted=()
            for episode in "${episodes[@]}"; do
                # Check if episode is not in failed list
                is_failed=false
                for failed_ep in "${failed_episodes[@]}"; do
                    if [ "$episode" = "$failed_ep" ]; then
                        is_failed=true
                        break
                    fi
                done
                if [ "$is_failed" = false ]; then
                    successfully_deleted+=("$episode")
                fi
            done
            
            if [ ${#successfully_deleted[@]} -gt 0 ]; then
                success_episodes_list=$(printf '%s,' "${successfully_deleted[@]}")
                success_episodes_list=${success_episodes_list%,}
                
                # Create temporary Python script
                cat > update_state_temp.py << EOF
import json
try:
    with open('processing_state.json', 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    processed = state.get('processed_episodes', [])
    episodes_to_remove = [$success_episodes_list]
    
    # Remove episodes from processed list
    removed_count = 0
    for ep in episodes_to_remove:
        if ep in processed:
            processed.remove(ep)
            removed_count += 1
    
    # Remove episode ranges
    ranges = state.get('episode_speaker_ranges', {})
    ranges_removed = 0
    for ep in episodes_to_remove:
        ep_str = str(ep)
        if ep_str in ranges:
            del ranges[ep_str]
            ranges_removed += 1
    
    state['processed_episodes'] = processed
    state['episode_speaker_ranges'] = ranges
    
    with open('processing_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    if removed_count > 0:
        print(f'✅ 更新狀態記錄 (移除 {removed_count} 集)')
except Exception as e:
    print(f'❌ 更新狀態失敗: {e}')
EOF
                
                run_python_script update_state_temp.py
            fi
        fi
        
        echo ""
        if [ ${#failed_episodes[@]} -gt 0 ]; then
            echo "📊 清除結果: ✅ $success_count 集成功, ❌ ${#failed_episodes[@]} 集失敗"
        else
            echo "📊 清除結果: ✅ 成功清除 $success_count 集"
        fi
    else
        echo "取消操作"
    fi
    
    echo ""
    pause_for_input
}

# Function to clean all data (dangerous operation)
clean_all_data() {
    echo ""
    echo "💀 全部清除 (危險操作)"
    echo "=================="
    echo ""
    echo "⚠️  這將刪除所有處理過的資料，包括："
    echo "  📁 輸出檔案 (output/)"
    echo "  📁 切分資料集 (split_dataset/)"
    echo "  📄 處理狀態 (processing_state.json)"
    echo ""
    echo "🚨 這個操作無法復原！"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        if get_confirmation "再次確認：真的要刪除所有資料嗎？"; then
            echo ""
            echo "💀 全部清除中..."
            
            local success=true
            
            # Remove output directory
            if [ -d "output" ]; then
                if rm -rf output; then
                    echo "✅ 已清除輸出檔案"
                else
                    echo "❌ 清除輸出檔案失敗"
                    success=false
                fi
            fi
            
            # Remove split dataset
            if [ -d "split_dataset" ]; then
                if rm -rf split_dataset; then
                    echo "✅ 已清除切分資料集"
                else
                    echo "❌ 清除切分資料集失敗"
                    success=false
                fi
            fi
            
            # Remove processing state
            if [ -f "processing_state.json" ]; then
                if rm -f processing_state.json; then
                    echo "✅ 已清除處理狀態"
                else
                    echo "❌ 清除處理狀態失敗"
                    success=false
                fi
            fi
            
            if [ "$success" = true ]; then
                echo ""
                echo "💀 全部清除完成！"
            else
                echo ""
                echo "⚠️  部分清除失敗，請檢查權限"
            fi
        else
            echo "取消操作"
        fi
    else
        echo "取消操作"
    fi
    
    echo ""
    pause_for_input
}

# Main cleanup menu function
show_cleanup_menu() {
    while true; do
        echo ""
        echo "🧹 清理數據"
        echo "=========="
        echo "1. 清除所有輸出檔案 (保留狀態記錄)"
        echo "2. 重置處理狀態 (保留檔案)"
        echo "3. 清除切分資料集 (保留原始處理檔案)"
        echo "4. 清除特定集數處理結果 (檔案+狀態)"
        echo "5. 全部清除 (危險操作)"
        echo "6. 返回主選單"
        echo ""
        echo "💡 說明："
        echo "  • 選項 1-3: 部分清除，可分別管理處理檔案、狀態記錄、切分資料"
        echo "  • 選項 4: 完整清除指定集數的所有相關資料"
        echo "  • 選項 5: 清除所有資料，回到初始狀態"
        echo ""
        echo -n "請選擇 [1-6]: "
        read choice
        
        case "$choice" in
            1)
                clean_output_files
                ;;
            2)
                reset_processing_state
                ;;
            3)
                clean_split_dataset
                ;;
            4)
                clean_specific_episodes
                ;;
            5)
                clean_all_data
                ;;
            6)
                break
                ;;
            *)
                echo "❌ 無效選項，請重新選擇"
                pause_for_input
                ;;
        esac
    done
}