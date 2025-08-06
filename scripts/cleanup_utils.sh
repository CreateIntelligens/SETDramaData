#!/bin/bash

# Cleanup utilities for etl interactive script
# All data cleaning and removal functions

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_utils.sh"

# Function to clean all output files
clean_output_files() {
    echo ""
    echo "🗑️ 清除所有輸出檔案"
    echo "=================="
    
    # Use DEFAULT_PROCESSED_DIR from .env or fallback
    PROCESSED_DIR="${DEFAULT_PROCESSED_DIR:-data/output}"
    
    if [ ! -d "$PROCESSED_DIR" ]; then
        echo "❌ 輸出目錄不存在: $PROCESSED_DIR"
        pause_for_input
        return
    fi
    
    # Count files to be deleted
    wav_count=$(find "$PROCESSED_DIR" -name "*.wav" 2>/dev/null | wc -l)
    txt_count=$(find "$PROCESSED_DIR" -name "*.txt" 2>/dev/null | wc -l)
    tsv_count=$(find "$PROCESSED_DIR" -name "*.tsv" 2>/dev/null | wc -l)
    
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
        
        if rm -rf "$PROCESSED_DIR"; then
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
    
    local has_json=false
    local has_db=false
    
    if [ -f "processing_state.json" ]; then
        has_json=true
    fi
    
    local db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
    if [ -f "$db_path" ]; then
        has_db=true
    fi
    
    if [ "$has_json" = false ] && [ "$has_db" = false ]; then
        echo "❌ 沒有找到任何處理狀態檔案"
        pause_for_input
        return
    fi
    
    echo "📋 當前處理狀態："
    
    # Check SQLite database first
    if [ "$has_db" = true ]; then
        echo "🗄️ SQLite資料庫狀態:"
        local python_cmd=$(detect_python)
        if [ -n "$python_cmd" ]; then
            $python_cmd "src/speaker_db_manager.py" --database "$db_path" stats
        fi
        echo ""
    fi
    
    # Check legacy JSON if exists
    if [ "$has_json" = true ]; then
        echo "📄 舊版JSON狀態 (legacy):"
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
    
except Exception as e:
    print(f'無法讀取狀態檔案: {e}')
EOF
        
        run_python_script reset_status_temp.py
        echo ""
    fi
    
    echo ""
    echo "⚠️  這將重置所有處理狀態記錄，包括："
    if [ "$has_db" = true ]; then
        echo "  🗄️ SQLite資料庫 ($db_path)"
    fi
    if [ "$has_json" = true ]; then
        echo "  📄 舊版JSON狀態 (processing_state.json)"
    fi
    echo ""
    echo "💡 處理過的檔案不會被刪除"
    
    if get_confirmation "確定要重置所有狀態嗎？"; then
        echo ""
        echo "🔄 重置中..."
        
        local success=true
        
        # Remove SQLite database
        if [ "$has_db" = true ]; then
            if rm -f "$db_path"; then
                echo "✅ SQLite資料庫已重置"
            else
                echo "❌ SQLite資料庫重置失敗"
                success=false
            fi
        fi
        
        # Remove legacy JSON
        if [ "$has_json" = true ]; then
            if rm -f "processing_state.json"; then
                echo "✅ 舊版JSON狀態已重置"
            else
                echo "❌ 舊版JSON狀態重置失敗"
                success=false
            fi
        fi
        
        if [ "$success" = true ]; then
            echo "✅ 所有處理狀態已重置（檔案保留）"
        else
            echo "⚠️ 部分重置失敗"
        fi
    else
        echo "❌ 已取消"
    fi
    
    echo ""
    pause_for_input
}

# Function to clean split dataset
clean_split_dataset() {
    echo ""
    echo "🗑️ 清除切分資料集"
    echo "================"
    
    # Use DEFAULT_SPLIT_DIR from .env or fallback
    SPLIT_DIR="${DEFAULT_SPLIT_DIR:-data/split_dataset}"
    
    if [ ! -d "$SPLIT_DIR" ]; then
        echo "❌ 切分資料集目錄不存在: $SPLIT_DIR"
        pause_for_input
        return
    fi
    
    # Count files to be deleted with detailed breakdown
    train_count=$(find "$SPLIT_DIR/train" -name "*.wav" 2>/dev/null | wc -l)
    test_count=$(find "$SPLIT_DIR/test" -name "*.wav" 2>/dev/null | wc -l)
    train_txt_count=$(find "$SPLIT_DIR/train" -name "*.txt" 2>/dev/null | wc -l)
    test_txt_count=$(find "$SPLIT_DIR/test" -name "*.txt" 2>/dev/null | wc -l)
    
    # Count episodes in split dataset
    train_episodes=$(find "$SPLIT_DIR/train" -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | wc -l)
    test_episodes=$(find "$SPLIT_DIR/test" -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | wc -l)
    
    echo "📊 切分資料集狀態："
    echo "  訓練集: $train_count 音頻檔 + $train_txt_count 文字檔 ($train_episodes 集)"
    echo "  測試集: $test_count 音頻檔 + $test_txt_count 文字檔 ($test_episodes 集)"
    echo "  總計: $((train_count + test_count)) 音頻檔"
    echo ""
    echo "⚠️  這只會刪除切分資料集，不影響原始處理檔案 ($PROCESSED_DIR) 和狀態記錄！"
    echo "💡 原始處理檔案仍可重新切分"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        echo "🗑️ 刪除中..."
        
        if rm -rf "$SPLIT_DIR"; then
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
    
    # Use environment variables for output directory
    PROCESSED_DIR="${DEFAULT_PROCESSED_DIR:-data/output}"
    
    if ! check_directory "$PROCESSED_DIR"; then
        pause_for_input
        return
    fi
    
    # Show available episodes with detailed status
    echo "📋 當前狀態："
    
    # Check SQLite database first
    local python_cmd=$(detect_python)
    local db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
    if [ -f "$db_path" ] && [ -n "$python_cmd" ]; then
        echo "🗄️ SQLite資料庫:"
        $python_cmd "src/database_cleanup.py" --database "$db_path" show
        echo ""
    fi
    
    # Check legacy JSON if exists
    if [ -f "processing_state.json" ]; then
        echo "📄 舊版JSON狀態 (legacy):"
        
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
        if [ -d "$PROCESSED_DIR" ]; then
            echo "📁 $PROCESSED_DIR 目錄存在，檢查檔案..."
            episodes_in_output=$(find "$PROCESSED_DIR" -maxdepth 2 -type d -name "[0-9][0-9][0-9]" 2>/dev/null | sed 's/.*\///;s/^0*//' | sort -n | uniq)
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
            for speaker_dir in $PROCESSED_DIR/*/; do
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
        if [ "$success_count" -gt 0 ]; then
            
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
                # Update SQLite database
                local db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
                if [ -f "$db_path" ] && [ -n "$python_cmd" ]; then
                    echo "🗄️ 更新SQLite資料庫狀態..."
                    $python_cmd "src/database_cleanup.py" --database "$db_path" remove "${successfully_deleted[@]}"
                fi
                
                # Update legacy JSON if exists
                if [ -f "processing_state.json" ]; then
                    success_episodes_list=$(printf '%s,' "${successfully_deleted[@]}")
                    success_episodes_list=${success_episodes_list%,}
                    
                    echo "📄 更新舊版JSON狀態..."
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
        print(f'✅ 更新舊版JSON狀態 (移除 {removed_count} 集)')
except Exception as e:
    print(f'❌ 更新舊版JSON狀態失敗: {e}')
EOF
                    
                    run_python_script update_state_temp.py
                fi
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
    # Use environment variables for paths
    PROCESSED_DIR="${DEFAULT_PROCESSED_DIR:-data/output}"
    SPLIT_DIR="${DEFAULT_SPLIT_DIR:-data/split_dataset}"
    
    echo "⚠️  這將刪除所有處理過的資料，包括："
    echo "  📁 輸出檔案 ($PROCESSED_DIR)"
    echo "  📁 切分資料集 ($SPLIT_DIR)"
    local db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
    if [ -f "$db_path" ]; then
        echo "  🗄️ SQLite資料庫 ($db_path)"
    fi
    if [ -f "processing_state.json" ]; then
        echo "  📄 舊版JSON狀態 (processing_state.json)"
    fi
    echo ""
    echo "🚨 這個操作無法復原！"
    
    if get_confirmation "確定要繼續嗎？"; then
        echo ""
        if get_confirmation "再次確認：真的要刪除所有資料嗎？"; then
            echo ""
            echo "💀 全部清除中..."
            
            local success=true
            
            # Remove output directory
            if [ -d "$PROCESSED_DIR" ]; then
                if rm -rf "$PROCESSED_DIR"; then
                    echo "✅ 已清除輸出檔案"
                else
                    echo "❌ 清除輸出檔案失敗"
                    success=false
                fi
            fi
            
            # Remove split dataset
            if [ -d "$SPLIT_DIR" ]; then
                if rm -rf "$SPLIT_DIR"; then
                    echo "✅ 已清除切分資料集"
                else
                    echo "❌ 清除切分資料集失敗"
                    success=false
                fi
            fi
            
            # Remove SQLite database
            if [ -f "$db_path" ]; then
                if rm -f "$db_path"; then
                    echo "✅ 已清除SQLite資料庫"
                else
                    echo "❌ 清除SQLite資料庫失敗"
                    success=false
                fi
            fi
            
            # Remove legacy JSON
            if [ -f "processing_state.json" ]; then
                if rm -f processing_state.json; then
                    echo "✅ 已清除舊版JSON狀態"
                else
                    echo "❌ 清除舊版JSON狀態失敗"
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