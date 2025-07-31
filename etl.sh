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
source "scripts/smart_process.sh"

# Load management modules
source "scripts/model_management.sh"
source "scripts/settings_management.sh"
source "scripts/database_management.sh"
source "scripts/uvr5_utils.sh"


# Smart process menu
smart_process_menu() {
    echo ""
    echo "🚀 智慧一條龍處理"
    echo "=================="
    echo ""
    echo "💡 說明："
    echo "• 自動檢測集數是否已處理，如已處理會自動清理重做"
    echo "• 處理完成後立即切分該集，無需手動操作"
    echo "• 使用 .env 設定的路徑，無需重複詢問"
    echo "• 全程自動化，無需確認"
    echo ""
    echo "請選擇處理方式："
    echo "1. 🚀 智慧一條龍處理 (標準)"
    echo "2. 🎵 智慧一條龍處理 + UVR5人聲分離"
    echo "3. 處理單集"
    echo "4. 處理多集"
    echo "5. 返回主選單"
    echo ""
    echo -n "請選擇 [1-5]: "
    read choice
    
    case "$choice" in
        1)
            echo ""
            echo "📋 標準智慧一條龍處理"
            echo "請輸入集數範圍（例如：1 2 3 或 1-5）："
            echo -n "集數: "
            read episodes_input
            
            # Parse episodes input
            episodes_output=$(validate_episode_input "$episodes_input")
            if [ $? -eq 0 ]; then
                readarray -t episodes <<< "$episodes_output"
                smart_process_episodes "${episodes[@]}"
            else
                echo "$episodes_output"
            fi
            pause_for_input
            ;;
        2)
            echo ""
            echo "🎵 智慧一條龍處理 + UVR5人聲分離"
            echo "⚠️  此選項會在標準處理完成後，對切分資料集進行 UVR5 人聲分離（去背景音樂）"
            echo ""
            echo "請輸入集數範圍（例如：1 2 3 或 1-5）："
            echo -n "集數: "
            read episodes_input
            
            # Parse episodes input
            episodes_output=$(validate_episode_input "$episodes_input")
            if [ $? -eq 0 ]; then
                readarray -t episodes <<< "$episodes_output"
                
                # 先執行標準處理
                echo "🚀 執行標準智慧處理..."
                if smart_process_episodes "${episodes[@]}"; then
                    echo ""
                    echo "🎵 開始 UVR5 人聲分離..."
                    echo -n "是否對處理結果進行 UVR5 人聲分離? [Y/n]: "
                    read confirm_uvr5
                    
                    if [[ ! "$confirm_uvr5" =~ ^[Nn]$ ]]; then
                        # 檢查 UVR5 環境
                        if check_uvr5_environment >/dev/null 2>&1; then
                            uvr5_enhance_split_dataset "data/split_dataset" "false"
                        else
                            echo "❌ UVR5 環境未準備就緒"
                            echo "請先檢查 UVR5 設定和模型檔案"
                        fi
                    fi
                else
                    echo "❌ 標準處理失敗，跳過 UVR5 人聲分離"
                fi
            else
                echo "$episodes_output"
            fi
            pause_for_input
            ;;
        3)
            echo ""
            echo -n "請輸入集數: "
            read episode_num
            if [[ "$episode_num" =~ ^[0-9]+$ ]]; then
                smart_process_episode "$episode_num"
            else
                echo "❌ 無效集數"
            fi
            pause_for_input
            ;;
        4)
            echo ""
            echo "請輸入集數範圍（例如：1 2 3 或 1-5）："
            echo -n "集數: "
            read episodes_input
            
            # Parse episodes input
            episodes_output=$(validate_episode_input "$episodes_input")
            if [ $? -eq 0 ]; then
                readarray -t episodes <<< "$episodes_output"
                smart_process_episodes "${episodes[@]}"
            else
                echo "$episodes_output"
            fi
            pause_for_input
            ;;
        5)
            return
            ;;
        *)
            echo "❌ 無效選項"
            pause_for_input
            ;;
    esac
}

# Function to show main menu
show_menu() {
    clear
    echo "📊 Breeze ASR - ETL Pipeline (含 UVR5 人聲分離)"
    echo "============================================"
    echo ""
    echo "請選擇功能："
    echo "1. 🚀 智慧一條龍處理 (Smart Auto Process)"
    echo "2. 📥 處理集數 (Process Episodes)"
    echo "3. 🔄 處理並切分 (Process & Split)"
    echo "4. 📤 切分訓練/測試集 (Split Dataset)"
    echo "5. 📊 狀態查看 (View Status)"
    echo "6. 🧹 清理數據 (Clean Data)" 
    echo "7. 🤖 模型管理 (Model Management)"
    echo "8. ⚙️ 設定管理 (Settings)"
    echo "9. 🗄️ 資料庫管理 (Database Management)"
    echo "10. 🎵 UVR5 人聲分離 (去背景音) (UVR5 Vocal Separation)"
    echo "0. 離開 (Exit)"
    echo ""
    echo -n "請輸入選項 [0-10]: "
}

# Main loop
main() {
    while true; do
        show_menu
        read -r choice
        
        case "$choice" in
            1)
                smart_process_menu
                ;;
            2)
                process_episodes_menu
                ;;
            3)
                process_and_split_menu
                ;;
            4)
                split_dataset_menu
                ;;
            5)
                show_status
                ;;
            6)
                show_cleanup_menu
                ;;
            7)
                show_model_menu
                ;;
            8)
                show_settings_menu
                ;;
            9)
                show_database_menu
                ;;
            10)
                show_uvr5_menu
                ;;
            0)
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