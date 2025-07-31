#!/bin/bash

# =============================================================================
# UVR5 人聲分離工具函數
# =============================================================================
# 功能：針對切分後的短音檔進行人聲分離，去除背景音樂
# 作者：Breeze ASR ETL Pipeline
# 版本：1.0
# =============================================================================

# 載入通用工具函數
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# -----------------------------------------------------------------------------
# UVR5 配置參數
# -----------------------------------------------------------------------------
UVR5_MODEL_PATH="${UVR5_MODEL_PATH:-models/uvr5}"
UVR5_VOCAL_MODEL="${UVR5_VOCAL_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
UVR5_DEVICE="${UVR5_DEVICE:-auto}"
UVR5_BATCH_SIZE="${UVR5_BATCH_SIZE:-1}"

# -----------------------------------------------------------------------------
# 核心函數
# -----------------------------------------------------------------------------

# 檢查 UVR5 環境是否完整
check_uvr5_environment() {
    echo "🔍 檢查 UVR5 環境..."
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    # 檢查模型檔案
    local model_file="$UVR5_MODEL_PATH/$UVR5_VOCAL_MODEL"
    if [ -f "$model_file" ]; then
        local model_size=$(du -h "$model_file" | cut -f1)
        echo "✅ UVR5 模型檔案存在: $UVR5_VOCAL_MODEL ($model_size)"
    else
        echo "❌ UVR5 模型檔案不存在: $model_file"
        return 1
    fi
    
    # 檢查 Python 依賴
    echo "📦 檢查 Python 依賴..."
    if $python_cmd -c "
import sys
sys.path.append('src')

try:
    from uvr5_processor import UVR5Processor
    processor = UVR5Processor()
    model_info = processor.get_model_info()
    
    print('✅ UVR5 處理器可用')
    print(f'🎮 設備: {model_info[\"device\"]}')
    print(f'📁 模型路徑: {model_info[\"model_path\"]}')
    print(f'📊 批次大小: {model_info[\"batch_size\"]}')
    
    processor.cleanup()
    
except ImportError as e:
    print(f'❌ 依賴套件缺失: {e}')
    exit(1)
except Exception as e:
    print(f'❌ UVR5 環境檢查失敗: {e}')
    exit(1)
" 2>/dev/null; then
        echo "✅ UVR5 環境檢查完成"
        return 0
    else
        echo "❌ UVR5 環境檢查失敗"
        return 1
    fi
}

# 對指定目錄進行 UVR5 批量人聲分離處理
uvr5_enhance_directory() {
    local input_dir="$1"
    local backup_original="${2:-false}"
    
    if [ -z "$input_dir" ]; then
        echo "❌ 請提供輸入目錄"
        echo "用法: uvr5_enhance_directory <目錄路徑> [backup_original]"
        return 1
    fi
    
    if [ ! -d "$input_dir" ]; then
        echo "❌ 目錄不存在: $input_dir"
        return 1
    fi
    
    echo "🎵 開始對目錄進行 UVR5 人聲分離..."
    echo "📁 目錄: $input_dir"
    echo "💾 備份原檔: $backup_original"
    echo ""
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    # 執行批量處理
    if $python_cmd -c "
import sys
sys.path.append('src')

from uvr5_processor import UVR5Processor

try:
    processor = UVR5Processor(
        model_path='$UVR5_MODEL_PATH',
        vocal_model='$UVR5_VOCAL_MODEL',
        device='$UVR5_DEVICE',
        batch_size=$UVR5_BATCH_SIZE
    )
    
    result = processor.batch_enhance(
        input_dir='$input_dir',
        pattern='*.wav',
        backup_original=$backup_original
    )
    
    if result['success']:
        print(f'\\n✅ 目錄處理完成')
        print(f'📊 處理檔案: {result[\"processed_files\"]}/{result[\"total_files\"]}')
        if result['failed_files'] > 0:
            print(f'❌ 失敗檔案: {result[\"failed_files\"]}')
    else:
        print(f'❌ 目錄處理失敗: {result.get(\"error\", \"Unknown error\")}')
        exit(1)
    
finally:
    processor.cleanup()

" 2>/dev/null; then
        echo "✅ 目錄 UVR5 人聲分離完成"
        return 0
    else
        echo "❌ 目錄 UVR5 人聲分離失敗"
        return 1
    fi
}

# 對切分後的資料集進行 UVR5 人聲分離處理
uvr5_enhance_split_dataset() {
    local split_dir="${1:-data/split_dataset}"
    local backup_original="${2:-false}"
    
    if [ ! -d "$split_dir" ]; then
        echo "❌ 切分資料集目錄不存在: $split_dir"
        return 1
    fi
    
    echo "📊 開始對切分資料集進行 UVR5 人聲分離..."
    echo "📁 資料集目錄: $split_dir"
    echo "💾 備份原檔: $backup_original"
    echo ""
    
    # 檢查 UVR5 環境
    if ! check_uvr5_environment; then
        return 1
    fi
    
    local python_cmd=$(detect_python)
    
    # 執行切分資料集處理
    if $python_cmd -c "
import sys
sys.path.append('src')

from uvr5_processor import UVR5Processor

try:
    processor = UVR5Processor(
        model_path='$UVR5_MODEL_PATH',
        vocal_model='$UVR5_VOCAL_MODEL',
        device='$UVR5_DEVICE',
        batch_size=$UVR5_BATCH_SIZE
    )
    
    result = processor.enhance_split_dataset(
        split_dir='$split_dir',
        backup_original=$backup_original
    )
    
    if result['success']:
        print('\\n🎉 切分資料集 UVR5 人聲分離完成！')
    else:
        print('❌ 切分資料集 UVR5 人聲分離失敗')
        exit(1)
    
finally:
    processor.cleanup()

" 2>/dev/null; then
        echo ""
        echo "🎉 切分資料集 UVR5 人聲分離完成！"
        return 0
    else
        echo "❌ 切分資料集 UVR5 人聲分離失敗"
        return 1
    fi
}

# 測試 UVR5 單個音檔處理功能
uvr5_test_single_file() {
    local input_file="$1"
    
    if [ -z "$input_file" ]; then
        echo "❌ 請提供音檔路徑"
        echo "用法: uvr5_test_single_file <音檔路徑>"
        return 1
    fi
    
    if [ ! -f "$input_file" ]; then
        echo "❌ 音檔不存在: $input_file"
        return 1
    fi
    
    echo "🎵 測試 UVR5 單檔處理..."
    echo "📁 檔案: $(basename "$input_file")"
    echo ""
    
    local python_cmd=$(detect_python)
    if [ -z "$python_cmd" ]; then
        echo "❌ 找不到 Python"
        return 1
    fi
    
    # 創建測試輸出目錄
    local test_output_dir="data/uvr5_test_output"
    mkdir -p "$test_output_dir"
    
    local output_file="$test_output_dir/$(basename "$input_file" .wav)_enhanced.wav"
    
    # 執行單檔測試
    if $python_cmd -c "
import sys
sys.path.append('src')

from uvr5_processor import UVR5Processor

try:
    processor = UVR5Processor(
        model_path='$UVR5_MODEL_PATH',
        vocal_model='$UVR5_VOCAL_MODEL',
        device='$UVR5_DEVICE'
    )
    
    result = processor.enhance_audio(
        input_path='$input_file',
        output_path='$output_file',
        backup_original=False
    )
    
    if result['success']:
        print(f'✅ 檔案處理成功')
        print(f'📁 輸出檔案: $output_file')
        print(f'⏱️  處理時間: {result[\"processing_time\"]:.2f} 秒')
        print(f'💾 記憶體使用: {result[\"memory_usage_mb\"]:.1f} MB')
        print(f'🎵 音頻增強: {\"是\" if result[\"enhanced\"] else \"否\"}')
    else:
        print(f'❌ 檔案處理失敗: {result.get(\"error\", \"Unknown error\")}')
        exit(1)
    
finally:
    processor.cleanup()

" 2>/dev/null; then
        echo ""
        echo "✅ UVR5 單檔測試完成"
        echo "📁 測試輸出: $output_file"
        return 0
    else
        echo "❌ UVR5 單檔測試失敗"
        return 1
    fi
}

# 顯示 UVR5 系統配置和環境狀態
show_uvr5_status() {
    echo "⚙️ UVR5 人聲分離配置狀態"
    echo "==================="
    echo ""
    echo "📋 環境變數:"
    echo "  UVR5_MODEL_PATH: ${UVR5_MODEL_PATH:-models/uvr5}"
    echo "  UVR5_VOCAL_MODEL: ${UVR5_VOCAL_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
    echo "  UVR5_DEVICE: ${UVR5_DEVICE:-auto}"
    echo "  UVR5_BATCH_SIZE: ${UVR5_BATCH_SIZE:-1}"
    echo ""
    
    # 檢查模型檔案
    local model_file="$UVR5_MODEL_PATH/$UVR5_VOCAL_MODEL"
    if [ -f "$model_file" ]; then
        local model_size=$(du -h "$model_file" | cut -f1)
        echo "📁 模型檔案: ✅ 存在 ($model_size)"
    else
        echo "📁 模型檔案: ❌ 不存在 ($model_file)"
    fi
    
    # 檢查 Python 環境
    local python_cmd=$(detect_python)
    if [ -n "$python_cmd" ]; then
        echo "🐍 Python: ✅ 可用 ($($python_cmd --version 2>&1))"
        
        # 檢查套件
        if $python_cmd -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null; then
            echo "📦 PyTorch: ✅ 可用"
        else
            echo "📦 PyTorch: ❌ 不可用"
        fi
        
        if $python_cmd -c "from audio_separator.separator import Separator; print('✅ audio-separator 可用')" 2>/dev/null; then
            echo "📦 audio-separator: ✅ 可用"
        else
            echo "📦 audio-separator: ❌ 不可用 (請執行: pip install 'audio-separator[gpu]')"
        fi
    else
        echo "🐍 Python: ❌ 不可用"
    fi
    
    echo ""
    echo "💡 使用建議:"
    echo "  UVR5 人聲分離功能可直接使用，透過 ETL 選單選項 10 進入"
    echo "  功能：從混合音頻中分離出純淨人聲，去除背景音樂"
}

# -----------------------------------------------------------------------------
# 使用者介面
# -----------------------------------------------------------------------------

# UVR5 主選單介面
show_uvr5_menu() {
    while true; do
        echo ""
        echo "🎵 UVR5 人聲分離選單"
        echo "==================="
        echo ""
        echo "請選擇功能："
        echo "1. 📊 對切分資料集進行 UVR5 人聲分離"
        echo "2. 📁 對指定目錄進行 UVR5 人聲分離"
        echo "3. 🎵 測試單個音檔"
        echo "4. 🔍 檢查 UVR5 環境"
        echo "5. ⚙️  顯示 UVR5 狀態"
        echo "6. 返回主選單"
        echo ""
        echo -n "請選擇 [1-6]: "
        read choice
        
        case "$choice" in
            1)
                echo ""
                echo -n "請輸入切分資料集路徑 (預設: data/split_dataset): "
                read split_dir
                split_dir="${split_dir:-data/split_dataset}"
                
                echo -n "是否備份原始檔案? [y/N]: "
                read backup_choice
                local backup_original="false"
                if [[ "$backup_choice" =~ ^[Yy]$ ]]; then
                    backup_original="true"
                fi
                
                uvr5_enhance_split_dataset "$split_dir" "$backup_original"
                pause_for_input
                ;;
            2)
                echo ""
                echo -n "請輸入目錄路徑: "
                read input_dir
                if [ -z "$input_dir" ]; then
                    echo "❌ 目錄路徑不能為空"
                    pause_for_input
                    continue
                fi
                
                echo -n "是否備份原始檔案? [y/N]: "
                read backup_choice
                local backup_original="false"
                if [[ "$backup_choice" =~ ^[Yy]$ ]]; then
                    backup_original="true"
                fi
                
                uvr5_enhance_directory "$input_dir" "$backup_original"
                pause_for_input
                ;;
            3)
                echo ""
                echo -n "請輸入音檔路徑: "
                read audio_file
                if [ -z "$audio_file" ]; then
                    echo "❌ 音檔路徑不能為空"
                    pause_for_input
                    continue
                fi
                
                uvr5_test_single_file "$audio_file"
                pause_for_input
                ;;
            4)
                echo ""
                check_uvr5_environment
                pause_for_input
                ;;
            5)
                echo ""
                show_uvr5_status
                pause_for_input
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

# 如果直接執行此腳本
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    if [ $# -eq 0 ]; then
        show_uvr5_menu
    else
        case "$1" in
            "check")
                check_uvr5_environment
                ;;
            "enhance_dir")
                uvr5_enhance_directory "${@:2}"
                ;;
            "enhance_split")
                uvr5_enhance_split_dataset "${@:2}"
                ;;
            "test_file")
                uvr5_test_single_file "${@:2}"
                ;;
            "status")
                show_uvr5_status
                ;;
            *)
                echo "UVR5 工具腳本"
                echo "============="
                echo "用法："
                echo "  $0                    # 顯示選單"
                echo "  $0 check              # 檢查環境"
                echo "  $0 enhance_dir <dir>  # 增強目錄"
                echo "  $0 enhance_split      # 增強切分資料集"
                echo "  $0 test_file <file>   # 測試單檔"
                echo "  $0 status             # 顯示狀態"
                ;;
        esac
    fi
fi