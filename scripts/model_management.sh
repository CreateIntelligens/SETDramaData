#!/bin/bash

# Model Management Functions
# 模型管理功能模組

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
    
    $python_cmd "src/download_models_offline.py"
    
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
    
    # 優先檢查正規離線配置
    local config_file="$models_dir/config.yaml"
    local seg_model="$models_dir/pyannote_model_segmentation-3.0.bin"
    local emb_model="$models_dir/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
    
    if [ -f "$config_file" ] && [ -f "$seg_model" ] && [ -f "$emb_model" ]; then
        echo "🎯 正規離線配置:"
        
        # Check config file
        if [ -f "$config_file" ]; then
            echo "  ✅ 配置檔案 (config.yaml)"
            local config_size=$(ls -lh "$config_file" | awk '{print $5}')
            echo "     📄 大小: $config_size"
        fi
        
        # Check segmentation model
        if [ -f "$seg_model" ]; then
            echo "  ✅ 分割模型 (segmentation-3.0)"
            local seg_size=$(ls -lh "$seg_model" | awk '{print $5}')
            echo "     📄 大小: $seg_size"
        fi
        
        # Check embedding model
        if [ -f "$emb_model" ]; then
            echo "  ✅ 嵌入模型 (wespeaker-voxceleb-resnet34-LM)"
            local emb_size=$(ls -lh "$emb_model" | awk '{print $5}')
            echo "     📄 大小: $emb_size"
        fi
        
        # Total size of offline models
        local total_size=$(du -sh "$models_dir"/*.bin "$models_dir"/*.yaml 2>/dev/null | awk '{sum+=$1} END {print sum"M"}' 2>/dev/null || echo "未知")
        echo "  📏 模型總大小: $total_size"
        
        echo ""
        echo "🎯 系統狀態: ✅ 正規離線模式 (推薦)"
        
    else
        echo "⚠️ 正規離線配置不完整:"
        [ ! -f "$config_file" ] && echo "  ❌ 缺少: config.yaml"
        [ ! -f "$seg_model" ] && echo "  ❌ 缺少: pyannote_model_segmentation-3.0.bin"
        [ ! -f "$emb_model" ] && echo "  ❌ 缺少: pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
        echo ""
        
        # 回退檢查 HuggingFace 模型
        local hf_dir="$models_dir/huggingface"
        if [ -d "$hf_dir" ]; then
            echo "🤖 備用 HuggingFace 快取:"
            
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
            local emb_model_hf="$hf_dir/models--pyannote--embedding"
            if [ -d "$emb_model_hf" ]; then
                echo "  ✅ Speaker Embedding"
                local emb_files=$(find "$emb_model_hf" -name "*.yaml" -o -name "*.bin" | wc -l)
                echo "     📄 檔案數: $emb_files"
            else
                echo "  ❌ Speaker Embedding (缺失)"
            fi
            
            # Total size
            local hf_total_size=$(du -sh "$hf_dir" 2>/dev/null | cut -f1)
            echo "  📏 快取大小: ${hf_total_size:-未知}"
            
            echo ""
            echo "🌐 系統狀態: ⚠️ 使用 HuggingFace 快取 (備用模式)"
            
        else
            echo "🌐 系統狀態: ❌ 需要下載模型"
        fi
    fi
    
    echo ""
    echo "💡 建議: 使用「下載模型到專案」建立完整的正規離線配置"
    
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
    
    # Check if official offline method is available
    if [ -f "src/offline_pipeline.py" ] && [ -f "models/config.yaml" ]; then
        echo "🎯 使用官方正規離線方法測試..."
        echo ""
        
        $python_cmd src/offline_pipeline.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 官方離線方法測試成功!"
            echo "🚀 系統已準備好使用正規離線 Pipeline"
        else
            echo ""
            echo "⚠️ 官方方法測試失敗，嘗試備用方法..."
            
            # No fallback needed - official method should work
        fi
    else
        echo "⚠️ 正規離線配置不完整，使用備用方法..."
        echo "缺少檔案:"
        [ ! -f "src/offline_pipeline.py" ] && echo "  - src/offline_pipeline.py"
        [ ! -f "models/config.yaml" ] && echo "  - models/config.yaml" 
        echo ""
        
        echo "❌ 請先設定正規離線配置"
        echo "💡 執行「下載模型到專案」後會自動建立必要檔案"
    fi
    
    pause_for_input
}
