#!/bin/bash

# Breeze ASR - Interactive Menu (Modularized)
# 互動式選單介面 (模組化版本)

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Load utility modules
source "src/common_utils.sh"
source "src/status_utils.sh"
source "src/cleanup_utils.sh"
source "src/process_utils.sh"

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
    echo "5. 清理數據 (Clean Data)"
    echo "6. 離開 (Exit)"
    echo ""
    echo -n "請輸入選項 [1-6]: "
}

# Process episodes menu (now modularized)
# Function is defined in process_utils.sh

# Process and split menu (now modularized)
# Function is defined in process_utils.sh

# Split dataset menu (now modularized)
# Function is defined in process_utils.sh

# Main loop
main() {
    while true; do
        show_menu
        read choice
        
        case "$choice" in
            1)
                process_episodes_menu
                ;;
            2)
                process_and_split_menu
                ;;
            3)
                split_dataset_menu
                ;;
            4)
                show_status
                ;;
            5)
                show_cleanup_menu
                ;;
            6)
                echo ""
                echo "👋 再見！"
                exit 0
                ;;
            *)
                echo ""
                echo "❌ 無效選項，請重新選擇"
                pause_for_input
                ;;
        esac
    done
}

# Run main function
main