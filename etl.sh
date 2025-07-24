#!/bin/bash

# Breeze ASR - ETL Pipeline
# 音頻數據 ETL 處理管線 (Extract, Transform, Load)

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
source "scripts/common_utils.sh"
source "scripts/status_utils.sh"
source "scripts/cleanup_utils.sh"
source "scripts/process_utils.sh"

# Load management modules
source "scripts/model_management.sh"
source "scripts/settings_management.sh"
source "scripts/database_management.sh"

# Function to show main menu
show_menu() {
    clear
    echo "📊 Breeze ASR - ETL Pipeline"
    echo "============================"
    echo ""
    echo "請選擇功能："
    echo "1. 📥 Extract - 處理集數 (Process Episodes)"
    echo "2. 🔄 Transform - 處理並切分 (Process & Split)"
    echo "3. 📤 Load - 切分訓練/測試集 (Split Dataset)"
    echo "4. 📊 狀態查看 (View Status)"
    echo "5. 🧹 清理數據 (Clean Data)" 
    echo "6. 🤖 模型管理 (Model Management)"
    echo "7. ⚙️ 設定管理 (Settings)"
    echo "8. 🗄️ 資料庫管理 (Database Management)"
    echo "9. 離開 (Exit)"
    echo ""
    echo -n "請輸入選項 [1-9]: "
}

# Main loop
main() {
    while true; do
        show_menu
        read -r choice
        
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
                show_model_menu
                ;;
            7)
                show_settings_menu
                ;;
            8)
                show_database_menu
                ;;
            9)
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