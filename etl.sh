#!/bin/bash

#  TTS - ETL Pipeline
# 音頻數據 ETL 處理管線 (Extract, Transform, Load)

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure data directories exist and have correct permissions
mkdir -p data/temp data/output data/uvr5_test_output 2>/dev/null || true
# Try to fix permissions if possible
[ -w data/ ] || echo "⚠️  data/ 目錄權限問題，建議在 Host 執行: sudo chown -R 1000:1000 data/ output/"

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
source "scripts/timing_log.sh"

# Load management modules
source "scripts/model_management.sh"
source "scripts/settings_management.sh"
source "scripts/database_management.sh"
source "scripts/uvr5_utils.sh"


# Smart process menu - 簡化版本
smart_process_menu() {
    echo ""
    echo "🚀 智慧 UVR5 處理"
    echo "=================="
    echo ""
    echo "💡 說明："
    echo "• 自動 UVR5 人聲分離 + 智慧處理 + 切分資料集"
    echo "• 支援集數輸入、路徑輸入或萬用字元匹配"
    echo ""
    echo "請選擇輸入方式："
    echo "1. 📊 按集數處理 (例如: 1 2 3 或 1-5) [預設]"
    echo "2. 📁 按路徑處理 (例如: data/audio/ 或 *.wav)"
    echo "3. 返回主選單"
    echo ""
    echo -n "請選擇 [1-3，預設1]: "
    read choice
    
    # 如果沒有輸入，預設為1
    choice="${choice:-1}"
    
    case "$choice" in
        1)
            echo ""
            echo "📊 按集數 UVR5 智慧處理"
            echo "請輸入集數範圍（例如：1 2 3 或 1-5）："
            echo -n "集數: "
            read episodes_input
            
            # Parse episodes input
            episodes_output=$(validate_episode_input "$episodes_input")
            if [ $? -eq 0 ]; then
                readarray -t episodes <<< "$episodes_output"
                
                # 路徑設定階段
                echo ""
                echo "📁 路徑設定確認"
                echo "================="
                
                local input_dir="${DEFAULT_INPUT_DIR:-data/願望(音軌及字幕檔)}"
                local uvr5_output_dir="${UVR5_OUTPUT_DIR:-data/uvr5_separated}"
                local pyannote_output_dir="${DEFAULT_PROCESSED_DIR:-data/output}"
                local split_dir="${DEFAULT_SPLIT_DIR:-data/split_dataset}"
                
                echo "📥 輸入目錄: $input_dir"
                echo -n "修改輸入目錄 (回車使用預設): "
                read custom_input_dir
                if [ -n "$custom_input_dir" ]; then
                    input_dir="$custom_input_dir"
                fi
                
                echo "🎵 UVR5 輸出目錄: $uvr5_output_dir"
                echo -n "修改 UVR5 輸出目錄 (回車使用預設): "
                read custom_uvr5_dir
                if [ -n "$custom_uvr5_dir" ]; then
                    uvr5_output_dir="$custom_uvr5_dir"
                fi
                
                echo "🤖 pyannote 輸出目錄: $pyannote_output_dir"
                echo -n "修改 pyannote 輸出目錄 (回車使用預設): "
                read custom_pyannote_dir
                if [ -n "$custom_pyannote_dir" ]; then
                    pyannote_output_dir="$custom_pyannote_dir"
                fi
                
                echo "📊 切分資料集目錄: $split_dir"
                echo -n "修改切分目錄 (回車使用預設): "
                read custom_split_dir
                if [ -n "$custom_split_dir" ]; then
                    split_dir="$custom_split_dir"
                fi
                
                local speakers_db_path="${SPEAKERS_DATABASE_PATH:-data/speakers.db}"
                echo "🗄️ Speakers 資料庫: $speakers_db_path"
                echo -n "修改 Speakers 資料庫路徑 (回車使用預設): "
                read custom_db_path
                if [ -n "$custom_db_path" ]; then
                    speakers_db_path="$custom_db_path"
                fi
                
                echo ""
                echo "🔍 最終路徑設定："
                echo "  📥 原始音檔: $input_dir"
                echo "  🎵 UVR5 人聲分離輸出: $uvr5_output_dir"
                echo "  🤖 pyannote 處理輸出: $pyannote_output_dir (使用 UVR5 輸出作為輸入)"
                echo "  📊 最終切分資料集: $split_dir"
                echo "  🗄️ Speakers 資料庫: $speakers_db_path"
                echo ""
                echo "💡 處理流程: 原始音檔 → UVR5 人聲分離 → pyannote 處理 → 切分資料集"
                echo ""
                echo -n "確認路徑設定？[Y/n]: "
                read confirm_paths
                
                if [[ "$confirm_paths" =~ ^[Nn]$ ]]; then
                    echo "❌ 已取消"
                    pause_for_input
                    return
                fi
                
                echo "✅ 開始處理 ${#episodes[@]} 集..."
                echo ""
                
                # 設定環境變數供後續使用
                export CUSTOM_INPUT_DIR="$input_dir"
                export CUSTOM_UVR5_OUTPUT_DIR="$uvr5_output_dir"
                export CUSTOM_PYANNOTE_OUTPUT_DIR="$pyannote_output_dir"
                export CUSTOM_SPLIT_DIR="$split_dir"
                export CUSTOM_SPEAKERS_DATABASE_PATH="$speakers_db_path"
                
                # 暫時禁用 set -e 以避免批次處理中斷
                set +e
                smart_process_episodes_with_uvr5 "${episodes[@]}"
                # 重新啟用 set -e
                set -e
            else
                echo "$episodes_output"
            fi
            pause_for_input
            ;;
        2)
            echo ""
            echo "📁 按路徑 UVR5 智慧處理"
            echo "支援格式："
            echo "  • 目錄路徑: data/audio/"
            echo "  • 單一檔案: input.wav"
            echo "  • 萬用字元: backup_*.wav 或 **/*.mp3"
            echo ""
            echo "💡 完整流程：UVR5 人聲分離 → pyannote 處理 → 切分資料集"
            echo ""
            echo -n "請輸入路徑或模式: "
            read input_path
            
            if [ -z "$input_path" ]; then
                echo "❌ 請提供有效路徑"
                pause_for_input
                return
            fi
            
            echo ""
            echo "🚀 開始完整 UVR5 智慧處理流程..."
            echo "輸入: $input_path"
            
            # 檢查 UVR5 環境
            if ! check_uvr5_environment >/dev/null 2>&1; then
                echo "❌ UVR5 環境未準備就緒，請先檢查設定"
                pause_for_input
                return
            fi
            
            # Step 1: UVR5 人聲分離
            echo ""
            echo "🎵 Step 1: UVR5 人聲分離..."
            local threads="${UVR5_MAX_WORKERS:-2}"
            local python_cmd=$(detect_python)
            
            if [ -z "$python_cmd" ]; then
                echo "❌ 找不到 Python"
                pause_for_input
                return
            fi
            
            if $python_cmd "uvr5_cli.py" "$input_path" --threads "$threads" --backup; then
                echo "✅ UVR5 人聲分離完成"
            else
                echo "❌ UVR5 處理失敗，請檢查日誌"
                pause_for_input
                return
            fi
            
            # Step 2: 詢問是否需要 pyannote 處理和切分
            echo ""
            echo "🤖 Step 2: pyannote 處理 + 切分資料集"
            echo -n "是否繼續進行 pyannote 處理和切分資料集? [Y/n]: "
            read continue_processing
            
            if [[ "$continue_processing" =~ ^[Nn]$ ]]; then
                echo "✅ 僅完成 UVR5 人聲分離"
                echo "💡 提示：處理後的檔案已覆蓋原檔案，備份檔案為 .bak"
                pause_for_input
                return
            fi
            
            echo ""
            echo "⚠️  注意：按路徑處理目前不支援自動 pyannote 處理"
            echo "💡 建議：如需完整智慧處理，請："
            echo "   1. 將處理後的音檔放到集數目錄結構中"
            echo "   2. 使用選項 1 (按集數處理) 進行完整處理"
            echo "   3. 或使用選單選項 4 手動切分已處理的資料"
            echo ""
            echo "📁 UVR5 處理完成的檔案位置：$input_path (已覆蓋原檔案)"
            echo "🔄 備份檔案：原檔案名.bak"
            
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
}

# Function to show main menu
show_menu() {
    clear
    echo "📊  TTS - ETL Pipeline (含 UVR5 人聲分離)"
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
