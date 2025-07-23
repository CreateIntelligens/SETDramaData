#!/bin/bash

# SETVoicePrep - Interactive Menu (Modularized)
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

# Function to show main menu
show_menu() {
    clear
    echo "🎤 SETVoicePrep - Drama Voice Processing Tool"
    echo "=========================================="
    echo ""
    echo "請選擇功能："
    echo "1. 處理集數 (Process Episodes)"
    echo "2. 處理並切分 (Process & Split)"
    echo "3. 切分訓練/測試集 (Split Dataset)"
    echo "4. 查看狀態 (View Status)"
    echo "5. 清理數據 (Clean Data)"
    echo "6. 模型管理 (Model Management)"
    echo "7. 設定管理 (Settings)"
    echo "8. 資料庫管理 (Database Management)"
    echo "9. 離開 (Exit)"
    echo ""
    echo -n "請輸入選項 [1-9]: "
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

# Model management menu
show_model_menu() {
    while true; do
        echo ""
        echo "🤖 模型管理"
        echo "=========="
        echo "1. 下載模型到專案 (Download Models)"
        echo "2. 檢查模型狀態 (Check Model Status)"
        echo "3. 測試模型載入 (Test Model Loading)"
        echo "4. 返回主選單 (Back to Main Menu)"
        echo ""
        echo -n "請選擇 [1-4]: "
        read choice
        
        case "$choice" in
            1)
                download_models_to_project
                ;;
            2)
                check_model_status
                ;;
            3)
                test_model_loading
                ;;
            4)
                return
                ;;
            *)
                echo "❌ 無效選項"
                pause_for_input
                ;;
        esac
    done
}

# Settings menu
show_settings_menu() {
    while true; do
        echo ""
        echo "⚙️ 設定管理"
        echo "=========="
        echo "1. 查看目前設定 (View Current Settings)"
        echo "2. 設定Embedding參數 (Configure Embedding)"
        echo "3. 設定處理模式 (Configure Processing Mode)"
        echo "4. 設定分段模式 (Configure Segmentation Mode)"
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
                configure_embedding_settings
                ;;
            3)
                configure_processing_mode
                ;;
            4)
                configure_segmentation_mode
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

# Download models to project
download_models_to_project() {
    echo ""
    echo "📥 下載模型到專案"
    echo "================"
    
    # Check if models already exist
    if [ -d "models/huggingface" ] && [ -n "$(ls -A models/huggingface 2>/dev/null)" ]; then
        echo "⚠️ 模型目錄已存在且非空"
        echo "📁 位置: $(pwd)/models/huggingface"
        echo ""
        if ! get_confirmation "是否要重新下載模型？"; then
            echo "❌ 已取消"
            pause_for_input
            return
        fi
    fi
    
    # Check for HF token
    if [ -z "${HUGGINGFACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
        echo "❌ 錯誤: 需要設定HuggingFace token"
        echo ""
        echo "請執行以下步驟："
        echo "1. 到 https://huggingface.co/settings/tokens 建立token"
        echo "2. 到 https://huggingface.co/pyannote/embedding 授權存取"
        echo "3. 在 .env 檔案中設定 HUGGINGFACE_TOKEN=your_token"
        echo ""
        pause_for_input
        return
    fi
    
    echo "🔍 檢查Python環境..."
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo "📥 開始下載模型..."
    echo ""
    
    $python_cmd "src/download_models_local.py"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型下載完成!"
        echo "📁 模型位置: $(pwd)/models/"
        echo "🚀 現在可以離線使用了"
    else
        echo ""
        echo "❌ 模型下載失敗"
    fi
    
    pause_for_input
}

# Check model status
check_model_status() {
    echo ""
    echo "📊 模型狀態檢查"
    echo "==============="
    
    local models_dir="$(pwd)/models"
    
    if [ ! -d "$models_dir" ]; then
        echo "❌ 模型目錄不存在: $models_dir"
        echo "💡 請先執行「下載模型到專案」"
        pause_for_input
        return
    fi
    
    echo "📁 模型目錄: $models_dir"
    echo ""
    
    # Check HuggingFace models
    local hf_dir="$models_dir/huggingface"
    if [ -d "$hf_dir" ]; then
        echo "🤖 HuggingFace 模型:"
        
        # Check diarization model
        local diar_model="$hf_dir/models--pyannote--speaker-diarization-3.1"
        if [ -d "$diar_model" ]; then
            echo "  ✅ Speaker Diarization 3.1"
            local diar_files=$(find "$diar_model" -name "*.yaml" -o -name "*.bin" | wc -l)
            echo "     📄 檔案數: $diar_files"
        else
            echo "  ❌ Speaker Diarization 3.1 (缺失)"
        fi
        
        # Check embedding model
        local emb_model="$hf_dir/models--pyannote--embedding"
        if [ -d "$emb_model" ]; then
            echo "  ✅ Speaker Embedding"
            local emb_files=$(find "$emb_model" -name "*.yaml" -o -name "*.bin" | wc -l)
            echo "     📄 檔案數: $emb_files"
        else
            echo "  ❌ Speaker Embedding (缺失)"
        fi
        
        # Total size
        local total_size=$(du -sh "$hf_dir" 2>/dev/null | cut -f1)
        echo "  📏 總大小: ${total_size:-未知}"
        
    else
        echo "❌ HuggingFace模型目錄不存在"
    fi
    
    echo ""
    
    # Check if system can use local models
    if [ -d "$hf_dir" ] && [ -n "$(ls -A "$hf_dir" 2>/dev/null)" ]; then
        echo "🎯 系統狀態: 可使用本地模型 (離線模式)"
    else
        echo "🌐 系統狀態: 需要網路下載模型 (線上模式)"
    fi
    
    pause_for_input
}

# Test model loading
test_model_loading() {
    echo ""
    echo "🧪 測試模型載入"
    echo "==============="
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo "🔧 測試模型載入中..."
    echo ""
    
    # Create a simple test script
    cat > /tmp/test_models.py << 'EOF'
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    print("🤖 測試 PyTorch...")
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   🖥️ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    print("\n📡 測試 pyannote.audio...")
    from pyannote.audio import Pipeline, Model
    print("   ✅ pyannote.audio 匯入成功")
    
    # Check local models
    models_dir = project_root / "models"
    if models_dir.exists():
        print(f"\n📁 檢測到本地模型: {models_dir}")
        os.environ['HF_HOME'] = str(models_dir / "huggingface")
        os.environ['TORCH_HOME'] = str(models_dir / "torch")
        print("   🔧 設定使用本地模型")
    
    print("\n🎤 測試 Diarization 模型載入...")
    try:
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
        print("   ✅ Diarization 模型載入成功")
    except Exception as e:
        print(f"   ❌ Diarization 模型載入失敗: {e}")
        sys.exit(1)
    
    print("\n🔊 測試 Embedding 模型載入...")
    try:
        model = Model.from_pretrained('pyannote/embedding')
        print("   ✅ Embedding 模型載入成功")
    except Exception as e:
        print(f"   ❌ Embedding 模型載入失敗: {e}")
        sys.exit(1)
    
    print("\n🎉 所有模型測試通過!")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    sys.exit(1)
EOF
    
    $python_cmd /tmp/test_models.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型測試完成 - 系統運作正常!"
    else
        echo ""
        echo "❌ 模型測試失敗 - 請檢查設定"
    fi
    
    # Cleanup
    rm -f /tmp/test_models.py
    
    pause_for_input
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
    echo "  SIMILARITY_THRESHOLD: ${SIMILARITY_THRESHOLD:-0.50}"
    # Show segmentation mode (only Hybrid mode is supported)
    if [ "${USE_HYBRID_SEGMENTATION:-false}" = "true" ]; then
        segmentation_mode="Hybrid模式 (結合Diarization與字幕)"
    else
        segmentation_mode="未設定 (建議啟用Hybrid模式)"
    fi
    echo "  分段模式: $segmentation_mode"
    echo "  USE_HYBRID_SEGMENTATION: ${USE_HYBRID_SEGMENTATION:-false}"
    
    # Default directories
    echo ""
    echo "📁 預設目錄:"
    echo "  DEFAULT_INPUT_DIR: ${DEFAULT_INPUT_DIR:-D:/願望(音軌及字幕檔)}"
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

# Configure embedding settings
configure_embedding_settings() {
    echo ""
    echo "🎛️ 設定Speaker識別參數"
    echo "======================"
    
    echo "目前設定:"
    echo "  跨集識別閾值: ${SIMILARITY_THRESHOLD:-0.50}"
    echo "  語音活動閾值: ${VOICE_ACTIVITY_THRESHOLD:-0.6}"
    echo "  Embedding更新: ${UPDATE_SPEAKER_EMBEDDINGS:-true}"
    echo ""
    
    echo "設定說明:"
    echo "• 跨集識別閾值: 控制不同集間speaker識別 (0.0-1.0)"
    echo "  - 較低值 (0.7-0.8): 更容易識別為同一speaker"
    echo "  - 較高值 (0.9-0.95): 更嚴格的speaker識別"
    echo ""
    
    # Speaker threshold  
    echo -n "新的跨集識別閾值 [目前: ${SIMILARITY_THRESHOLD:-0.50}]: "
    read new_speaker_threshold
    if [ -n "$new_speaker_threshold" ]; then
        export SIMILARITY_THRESHOLD="$new_speaker_threshold"
        update_env_setting "SIMILARITY_THRESHOLD" "$new_speaker_threshold"
    fi
    
    # Voice activity threshold
    echo -n "新的語音活動閾值 [目前: ${VOICE_ACTIVITY_THRESHOLD:-0.6}]: "
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
    
    echo "選擇處理模式:"
    echo "1. 標準模式 - 有embedding驗證 (推薦)"
    echo "2. 快速模式 - 無embedding驗證"
    echo "3. 嚴格模式 - 高精度驗證"
    echo ""
    echo -n "請選擇 [1-3]: "
    read mode_choice
    
    case "$mode_choice" in
        1)
            export SIMILARITY_THRESHOLD="0.50"
            export VOICE_ACTIVITY_THRESHOLD="0.6"
            echo "✅ 設定為標準模式"
            ;;
        2)
            export SIMILARITY_THRESHOLD="0.40"
            export VOICE_ACTIVITY_THRESHOLD="0.5"
            echo "✅ 設定為寬鬆模式"
            ;;
        3)
            export SIMILARITY_THRESHOLD="0.65"
            export VOICE_ACTIVITY_THRESHOLD="0.7"
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

# Configure segmentation mode
configure_segmentation_mode() {
    echo ""
    echo "🔄 設定分段模式"
    echo "==============="
    
    echo "目前使用: Hybrid模式 (結合Diarization與字幕)"
    echo ""
    echo "模式說明:"
    echo "• Hybrid模式結合了pyannote diarization的精確說話人識別"
    echo "  與字幕時間軸的完整內容保障"
    echo "• 確保不會遺漏任何字幕內容"
    echo "• 提供最佳的說話人辨識精度"
    echo "• 已針對中文語音進行優化"
    echo ""
    
    if [ "${USE_HYBRID_SEGMENTATION:-false}" = "true" ]; then
        echo "✅ Hybrid模式已啟用"
    else
        echo "🔧 啟用Hybrid模式？"
        if get_confirmation "確定要設定為Hybrid模式嗎？"; then
            # Set hybrid mode and disable others
            update_env_setting "USE_STREAMING_SEGMENTATION" "false"
            update_env_setting "USE_SUBTITLE_DRIVEN" "false"
            update_env_setting "USE_HYBRID_SEGMENTATION" "true"
            export USE_STREAMING_SEGMENTATION="false"
            export USE_SUBTITLE_DRIVEN="false"
            export USE_HYBRID_SEGMENTATION="true"
            echo "✅ 已設定為Hybrid模式"
        else
            echo "❌ 已取消"
        fi
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
        grep -v "^SIMILARITY_THRESHOLD\|^VOICE_ACTIVITY_THRESHOLD" .env > .env.tmp
        mv .env.tmp .env
        echo "💾 已備份原設定到 .env.backup"
    fi
    
    # Reset environment variables
    unset SIMILARITY_THRESHOLD
    unset VOICE_ACTIVITY_THRESHOLD
    
    echo "✅ 已重置為預設值"
    echo "   識別閾值: 0.50"
    echo "   語音活動閾值: 0.6"
    
    pause_for_input
}

# Database management menu
show_database_menu() {
    while true; do
        echo ""
        echo "🗄️ 資料庫管理"
        echo "=============="
        echo "1. 查看資料庫統計 (Database Stats)"
        echo "2. 列出所有speaker (List All Speakers)"
        echo "3. 查看speaker詳細資訊 (Speaker Details)"
        echo "4. 查看集數speaker對應 (Episode Mapping)"
        echo "5. 匯出資料庫 (Export Database)"
        echo "6. 從JSON遷移 (Migrate from JSON)"
        echo "7. 備份資料庫 (Backup Database)"
        echo "8. 返回主選單 (Back to Main Menu)"
        echo ""
        echo -n "請選擇 [1-8]: "
        read choice
        
        case "$choice" in
            1)
                show_database_stats
                ;;
            2)
                list_all_speakers
                ;;
            3)
                show_speaker_details
                ;;
            4)
                show_episode_mapping
                ;;
            5)
                export_database
                ;;
            6)
                migrate_from_json_menu
                ;;
            7)
                backup_database
                ;;
            8)
                return
                ;;
            *)
                echo "❌ 無效選項"
                pause_for_input
                ;;
        esac
    done
}

# Database management functions
show_database_stats() {
    echo ""
    echo "📊 資料庫統計"
    echo "============"
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" stats
    pause_for_input
}

list_all_speakers() {
    echo ""
    echo "👥 所有Speaker"
    echo "============="
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" list
    pause_for_input
}

show_speaker_details() {
    echo ""
    echo "👤 Speaker詳細資訊"
    echo "=================="
    echo -n "請輸入Speaker ID: "
    read speaker_id
    
    if [ -z "$speaker_id" ]; then
        echo "❌ 請輸入有效的Speaker ID"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" speaker "$speaker_id"
    pause_for_input
}

show_episode_mapping() {
    echo ""
    echo "📺 集數Speaker對應"
    echo "=================="
    echo -n "請輸入集數: "
    read episode_num
    
    if [ -z "$episode_num" ]; then
        echo "❌ 請輸入有效的集數"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" episode "$episode_num"
    pause_for_input
}

export_database() {
    echo ""
    echo "📤 匯出資料庫"
    echo "============"
    echo -n "輸出檔案名稱 [預設: speakers_backup.json]: "
    read output_file
    output_file="${output_file:-speakers_backup.json}"
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" export "$output_file"
    pause_for_input
}

migrate_from_json_menu() {
    echo ""
    echo "🔄 從JSON遷移"
    echo "============="
    echo "此功能會將舊的processing_state.json轉換為SQLite資料庫"
    echo ""
    
    # Check if legacy JSON exists
    local json_file="processing_state.json"
    if [ ! -f "$json_file" ]; then
        echo "❌ 找不到JSON檔案: $json_file"
        pause_for_input
        return
    fi
    
    echo "找到JSON檔案: $json_file"
    if ! get_confirmation "確定要開始遷移嗎？"; then
        echo "❌ 已取消"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    echo ""
    $python_cmd "src/speaker_db_manager.py" migrate "$json_file"
    pause_for_input
}

backup_database() {
    echo ""
    echo "💾 備份資料庫"
    echo "============"
    
    if [ ! -f "speakers.db" ]; then
        echo "❌ 找不到資料庫檔案: speakers.db"
        pause_for_input
        return
    fi
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到Python"
        pause_for_input
        return
    fi
    
    $python_cmd "src/speaker_db_manager.py" backup
    pause_for_input
}

# Run main function
main