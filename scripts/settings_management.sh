#!/bin/bash

# Settings Management Functions
# 設定管理功能模組

# Load utility functions
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# Function to update or add setting in .env file
update_env_setting() {
    local key="$1"
    local value="$2"
    local env_file=".env"
    
    # Create .env if it doesn't exist
    touch "$env_file"
    
    # Remove existing line with this key
    if grep -q "^${key}=" "$env_file"; then
        # Key exists, update it - use more robust pattern matching
        if command -v sed &> /dev/null; then
            # Use sed with proper escaping
            sed -i.bak "/^${key}=/c\\${key}=${value}" "$env_file"
            rm -f "${env_file}.bak"
        else
            # Fallback method
            grep -v "^${key}=" "$env_file" > "${env_file}.tmp"
            echo "${key}=${value}" >> "${env_file}.tmp"
            mv "${env_file}.tmp" "$env_file"
        fi
    else
        # Key doesn't exist, add it
        echo "${key}=${value}" >> "$env_file"
    fi
}

# Settings menu
show_settings_menu() {
    while true; do
        echo ""
        echo "⚙️ 設定管理"
        echo "=========="
        echo "1. 查看目前設定 (View Current Settings)"
        echo "2. 設定目錄路徑 (Configure Directory Paths)"
        echo "3. 設定Embedding參數 (Configure Embedding)"
        echo "4. 設定處理模式 (Configure Processing Mode)"
        echo "5. 重置為預設值 (Reset to Defaults)"
        echo "6. 返回主選單 (Back to Main Menu)"
        echo ""
        echo -n "請選擇 [1-6]: "
        read choice
        
        case "$choice" in
            1)
                show_current_settings
                ;;
            2)
                configure_directory_paths
                ;;
            3)
                configure_embedding_settings
                ;;
            4)
                configure_processing_mode
                ;;
            5)
                reset_to_defaults
                ;;
            6)
                return
                ;;
            *)
                echo "❌ 無效選項"
                pause_for_input
                ;;
        esac
    done
}

# Show current settings
show_current_settings() {
    echo ""
    echo "📋 目前設定"
    echo "=========="
    
    # Environment variables
    echo "🔧 環境變數:"
    echo "  HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:-未設定}"
    echo "  HF_TOKEN: ${HF_TOKEN:-未設定}"
    
    # Processing settings from .env or defaults
    echo ""
    echo "⚙️ 處理設定:"
    echo "  SIMILARITY_THRESHOLD: ${SIMILARITY_THRESHOLD:-0.40}"
    echo "  VOICE_ACTIVITY_THRESHOLD: ${VOICE_ACTIVITY_THRESHOLD:-0.1}"
    
    # Show speaker-level settings
    echo "  分段模式: 說話人級別模式 (唯一模式)"
    echo "  MIN_SPEAKER_DURATION: ${MIN_SPEAKER_DURATION:-5.0}秒"
    
    # Default directories
    echo ""
    echo "📁 預設目錄:"
    echo "  DEFAULT_INPUT_DIR: ${DEFAULT_INPUT_DIR:-願望(音軌及字幕檔)}"
    echo "  DEFAULT_PROCESSED_DIR: ${DEFAULT_PROCESSED_DIR:-output}"
    echo "  DEFAULT_SPLIT_DIR: ${DEFAULT_SPLIT_DIR:-split_dataset}"
    echo "  DEFAULT_TEST_RATIO: ${DEFAULT_TEST_RATIO:-0.2}"
    
    # Model status
    echo ""
    echo "🤖 模型狀態:"
    if [ -d "models/huggingface" ] && [ -n "$(ls -A models/huggingface 2>/dev/null)" ]; then
        echo "  本地模型: ✅ 已安裝"
    else
        echo "  本地模型: ❌ 未安裝"
    fi
    
    pause_for_input
}

# Configure directory paths
configure_directory_paths() {
    echo ""
    echo "📁 設定目錄路徑"
    echo "==============="
    
    echo "目前設定:"
    echo "  輸入目錄: ${DEFAULT_INPUT_DIR:-data/願望(音軌及字幕檔)}"
    echo "  處理結果目錄: ${DEFAULT_PROCESSED_DIR:-data/output}"
    echo "  切分資料集目錄: ${DEFAULT_SPLIT_DIR:-data/split_dataset}"
    echo "  測試集比例: ${DEFAULT_TEST_RATIO:-0.2}"
    echo ""
    
    echo "設定說明:"
    echo "• 輸入目錄: 存放音檔和字幕檔的目錄"
    echo "• 處理結果目錄: 分段後音檔的儲存位置"
    echo "• 切分資料集目錄: 訓練/測試集的儲存位置"
    echo "• 測試集比例: 用於測試的資料比例 (0.1-0.3)"
    echo ""
    
    # Input directory
    echo -n "輸入目錄 [目前: ${DEFAULT_INPUT_DIR:-data/願望(音軌及字幕檔)}]: "
    read new_input_dir
    if [ -n "$new_input_dir" ]; then
        export DEFAULT_INPUT_DIR="$new_input_dir"
        update_env_setting "DEFAULT_INPUT_DIR" "\"$new_input_dir\""
    fi
    
    # Processed directory
    echo -n "處理結果目錄 [目前: ${DEFAULT_PROCESSED_DIR:-data/output}]: "
    read new_processed_dir
    if [ -n "$new_processed_dir" ]; then
        export DEFAULT_PROCESSED_DIR="$new_processed_dir"
        update_env_setting "DEFAULT_PROCESSED_DIR" "\"$new_processed_dir\""
    fi
    
    # Split dataset directory
    echo -n "切分資料集目錄 [目前: ${DEFAULT_SPLIT_DIR:-data/split_dataset}]: "
    read new_split_dir
    if [ -n "$new_split_dir" ]; then
        export DEFAULT_SPLIT_DIR="$new_split_dir"
        update_env_setting "DEFAULT_SPLIT_DIR" "\"$new_split_dir\""
    fi
    
    # Test ratio
    echo -n "測試集比例 [目前: ${DEFAULT_TEST_RATIO:-0.2}]: "
    read new_test_ratio
    if [ -n "$new_test_ratio" ]; then
        export DEFAULT_TEST_RATIO="$new_test_ratio"
        update_env_setting "DEFAULT_TEST_RATIO" "\"$new_test_ratio\""
    fi
    
    echo ""
    echo "✅ 目錄設定已更新並儲存到 .env"
    echo "💡 新設定將在下次執行時生效"
    pause_for_input
}

# Configure embedding settings
configure_embedding_settings() {
    echo ""
    echo "🎛️ 設定Speaker識別參數"
    echo "======================"
    
    echo "目前設定:"
    echo "  跨集識別閾值: ${SIMILARITY_THRESHOLD:-0.40}"
    echo "  語音活動閾值: ${VOICE_ACTIVITY_THRESHOLD:-0.1}"
    echo ""
    
    echo "設定說明:"
    echo "• 跨集識別閾值: 控制不同集間speaker識別 (0.0-1.0)"
    echo "  - 較低值 (0.30-0.35): 更容易識別為同一speaker"
    echo "  - 較高值 (0.45-0.50): 更嚴格的speaker識別"
    echo "• 語音活動閾值: 控制語音活動檢測 (0.0-1.0)"
    echo "  - 較低值: 更敏感的語音檢測"
    echo "  - 較高值: 更嚴格的語音檢測"
    echo ""
    
    # Speaker threshold  
    echo -n "新的跨集識別閾值 [目前: ${SIMILARITY_THRESHOLD:-0.40}]: "
    read new_speaker_threshold
    if [ -n "$new_speaker_threshold" ]; then
        export SIMILARITY_THRESHOLD="$new_speaker_threshold"
        update_env_setting "SIMILARITY_THRESHOLD" "$new_speaker_threshold"
    fi
    
    # Voice activity threshold
    echo -n "新的語音活動閾值 [目前: ${VOICE_ACTIVITY_THRESHOLD:-0.1}]: "
    read new_vad_threshold
    if [ -n "$new_vad_threshold" ]; then
        export VOICE_ACTIVITY_THRESHOLD="$new_vad_threshold"
        update_env_setting "VOICE_ACTIVITY_THRESHOLD" "$new_vad_threshold"
    fi
    
    echo ""
    echo "✅ 設定已更新並儲存到 .env"
    pause_for_input
}

# Configure processing mode
configure_processing_mode() {
    echo ""
    echo "🏃 設定處理模式"
    echo "==============="
    
    local current_threshold="${SIMILARITY_THRESHOLD:-0.40}"
    local current_mode
    case "$current_threshold" in
        "0.40") current_mode="標準模式" ;;
        "0.35") current_mode="寬鬆模式" ;;
        "0.45") current_mode="嚴格模式" ;;
        *) current_mode="自訂模式 (閾值: $current_threshold)" ;;
    esac
    
    echo "目前模式: $current_mode"
    echo ""
    echo "選擇處理模式:"
    echo "1. 標準模式 - 適中的識別精度 (推薦)"
    echo "2. 寬鬆模式 - 更容易識別為同一人"
    echo "3. 嚴格模式 - 更嚴格的識別標準"
    echo ""
    echo -n "請選擇 [1-3]: "
    read mode_choice
    
    case "$mode_choice" in
        1)
            export SIMILARITY_THRESHOLD="0.40"
            export VOICE_ACTIVITY_THRESHOLD="0.1"
            echo "✅ 設定為標準模式"
            ;;
        2)
            export SIMILARITY_THRESHOLD="0.35"
            export VOICE_ACTIVITY_THRESHOLD="0.05"
            echo "✅ 設定為寬鬆模式"
            ;;
        3)
            export SIMILARITY_THRESHOLD="0.45"
            export VOICE_ACTIVITY_THRESHOLD="0.15"
            echo "✅ 設定為嚴格模式"
            ;;
        *)
            echo "❌ 無效選項"
            pause_for_input
            return
            ;;
    esac
    
    # Save to .env using update function
    update_env_setting "SIMILARITY_THRESHOLD" "$SIMILARITY_THRESHOLD"
    update_env_setting "VOICE_ACTIVITY_THRESHOLD" "$VOICE_ACTIVITY_THRESHOLD"
    
    echo "💾 設定已儲存到 .env"
    pause_for_input
}

# Configure speaker duration
configure_speaker_duration() {
    echo ""
    echo "⏱️ 設定最小說話人時長"
    echo "===================="
    
    echo "目前設定: ${MIN_SPEAKER_DURATION:-5.0}秒"
    echo ""
    echo "說明:"
    echo "• 最小說話人時長決定一個說話人需要說話多久才會被識別"
    echo "• 較低值 (3.0-4.0): 識別更多短發言者"
    echo "• 較高值 (6.0-8.0): 只識別主要角色"
    echo ""
    
    echo -n "新的最小說話人時長 (秒) [目前: ${MIN_SPEAKER_DURATION:-5.0}]: "
    read new_duration
    if [ -n "$new_duration" ]; then
        export MIN_SPEAKER_DURATION="$new_duration"
        update_env_setting "MIN_SPEAKER_DURATION" "$new_duration"
        echo "✅ 已設定最小說話人時長為 ${new_duration}秒"
    else
        echo "❌ 已取消"
    fi
    
    echo "💾 設定已儲存到 .env"
    pause_for_input
}

# Reset to defaults
reset_to_defaults() {
    echo ""
    echo "🔄 重置為預設值"
    echo "================"
    
    if ! get_confirmation "確定要重置所有設定為預設值嗎？"; then
        echo "❌ 已取消"
        pause_for_input
        return
    fi
    
    # Remove custom settings from .env
    if [ -f ".env" ]; then
        cp .env .env.backup
        grep -v "^SIMILARITY_THRESHOLD\|^VOICE_ACTIVITY_THRESHOLD\|^MIN_SPEAKER_DURATION\|^DEFAULT_INPUT_DIR\|^DEFAULT_PROCESSED_DIR\|^DEFAULT_SPLIT_DIR\|^DEFAULT_TEST_RATIO" .env > .env.tmp
        mv .env.tmp .env
        echo "💾 已備份原設定到 .env.backup"
    fi
    
    # Reset environment variables
    unset SIMILARITY_THRESHOLD
    unset VOICE_ACTIVITY_THRESHOLD
    unset MIN_SPEAKER_DURATION
    unset DEFAULT_INPUT_DIR
    unset DEFAULT_PROCESSED_DIR
    unset DEFAULT_SPLIT_DIR
    unset DEFAULT_TEST_RATIO
    
    echo "✅ 已重置為預設值"
    echo "   識別閾值: 0.40"
    echo "   語音活動閾值: 0.1"
    echo "   最小說話人時長: 5.0秒"
    echo "   輸入目錄: data/願望(音軌及字幕檔)"
    echo "   處理結果目錄: data/output"
    echo "   切分資料集目錄: data/split_dataset"
    echo "   測試集比例: 0.2"
    
    pause_for_input
}