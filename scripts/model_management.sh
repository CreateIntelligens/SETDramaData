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
        echo ""
        echo "📊 pyannote 模型:"
        echo "1. 下載 pyannote 模型 (Download pyannote Models)"
        echo "2. 檢查 pyannote 模型狀態 (Check pyannote Model Status)"
        echo "3. 測試 pyannote 模型載入 (Test pyannote Model Loading)"
        echo ""
        echo "🎵 UVR5 模型:"
        echo "4. 下載 UVR5 模型 (Download UVR5 Models)"
        echo "5. 檢查 UVR5 模型狀態 (Check UVR5 Model Status)"
        echo ""
        echo "6. 返回主選單 (Back to Main Menu)"
        echo ""
        echo -n "請選擇 [1-6]: "
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
                download_uvr5_models
                ;;
            5)
                check_uvr5_model_status
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

# Download UVR5 models
download_uvr5_models() {
    echo ""
    echo "📥 下載 UVR5 模型"
    echo "================"
    
    local uvr5_models_dir="models/uvr5"
    echo "📁 UVR5 模型目錄: $uvr5_models_dir"
    
    # Create UVR5 models directory
    mkdir -p "$uvr5_models_dir"
    
    # Check if default model already exists
    local default_model="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    local model_path="$uvr5_models_dir/$default_model"
    
    if [ -f "$model_path" ]; then
        local model_size=$(ls -lh "$model_path" | awk '{print $5}')
        echo "⚠️ 預設模型已存在: $default_model ($model_size)"
        echo ""
        if ! get_confirmation "是否要重新下載模型？"; then
            echo "❌ 已取消"
            pause_for_input
            return
        fi
    fi
    
    echo ""
    echo "🎵 可用的 UVR5 模型："
    echo "1. model_bs_roformer_ep_317_sdr_12.9755.ckpt (推薦 - 高品質人聲分離)"
    echo "2. Kim_Vocal_2.onnx (輕量級模型)"
    echo "3. UVR-MDX-NET-Voc_FT.onnx (另一個選擇)"
    echo ""
    echo -n "請選擇要下載的模型 [1-3] (預設: 1): "
    read model_choice
    
    case "${model_choice:-1}" in
        1)
            local selected_model="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            local download_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            ;;
        2)
            local selected_model="Kim_Vocal_2.onnx"
            local download_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx"
            ;;
        3)
            local selected_model="UVR-MDX-NET-Voc_FT.onnx"
            local download_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
            ;;
        *)
            echo "❌ 無效選擇，使用預設模型"
            local selected_model="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            local download_url="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            ;;
    esac
    
    echo ""
    echo "📥 開始下載: $selected_model"
    echo "🌐 下載來源: GitHub TRvlvr/model_repo"
    echo ""
    
    # Download with curl or wget
    local target_path="$uvr5_models_dir/$selected_model"
    
    if command -v curl &> /dev/null; then
        echo "🔄 使用 curl 下載..."
        if curl -L -o "$target_path" "$download_url" --progress-bar; then
            echo "✅ 下載完成!"
        else
            echo "❌ curl 下載失敗"
            pause_for_input
            return
        fi
    elif command -v wget &> /dev/null; then
        echo "🔄 使用 wget 下載..."
        if wget -O "$target_path" "$download_url" --progress=bar; then
            echo "✅ 下載完成!"
        else
            echo "❌ wget 下載失敗"
            pause_for_input
            return
        fi
    else
        echo "❌ 錯誤: 找不到 curl 或 wget 下載工具"
        echo "請手動下載模型到: $target_path"
        echo "下載網址: $download_url"
        pause_for_input
        return
    fi
    
    # Verify download
    if [ -f "$target_path" ]; then
        local file_size=$(ls -lh "$target_path" | awk '{print $5}')
        echo "📄 檔案大小: $file_size"
        
        # Update .env with downloaded model
        update_env_setting "UVR5_VOCAL_MODEL" "$selected_model"
        echo "⚙️ 已更新 .env 設定: UVR5_VOCAL_MODEL=$selected_model"
        
        echo ""
        echo "🎉 UVR5 模型下載並設定完成!"
        echo "📁 模型位置: $target_path"
        echo "🚀 現在可以使用 UVR5 人聲分離功能了"
    else
        echo "❌ 下載驗證失敗，檔案不存在"
    fi
    
    pause_for_input
}

# Check UVR5 model status
check_uvr5_model_status() {
    echo ""
    echo "📊 UVR5 模型狀態檢查"
    echo "==================="
    
    local uvr5_models_dir="models/uvr5"
    echo "📁 UVR5 模型目錄: $uvr5_models_dir"
    
    if [ ! -d "$uvr5_models_dir" ]; then
        echo "❌ UVR5 模型目錄不存在"
        echo "💡 請先執行「下載 UVR5 模型」"
        pause_for_input
        return
    fi
    
    echo ""
    echo "🎵 已安裝的 UVR5 模型："
    
    local model_count=0
    local current_model="${UVR5_VOCAL_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
    
    for model_file in "$uvr5_models_dir"/*.ckpt "$uvr5_models_dir"/*.onnx; do
        if [ -f "$model_file" ]; then
            local model_name=$(basename "$model_file")
            local model_size=$(ls -lh "$model_file" | awk '{print $5}')
            
            if [ "$model_name" = "$current_model" ]; then
                echo "  ✅ $model_name ($model_size) ⭐ 目前使用"
            else
                echo "  📄 $model_name ($model_size)"
            fi
            
            ((model_count++))
        fi
    done
    
    if [ $model_count -eq 0 ]; then
        echo "  ❌ 沒有找到任何 UVR5 模型檔案"
        echo ""
        echo "💡 建議："
        echo "  1. 執行「下載 UVR5 模型」自動下載"
        echo "  2. 或手動下載模型到 $uvr5_models_dir/"
    else
        echo ""
        echo "📈 統計："
        echo "  模型總數: $model_count"
        
        local total_size=$(du -sh "$uvr5_models_dir" 2>/dev/null | cut -f1)
        echo "  目錄大小: ${total_size:-未知}"
    fi
    
    echo ""
    echo "⚙️ 目前設定："
    echo "  UVR5_MODEL_PATH: ${UVR5_MODEL_PATH:-models/uvr5}"
    echo "  UVR5_VOCAL_MODEL: ${UVR5_VOCAL_MODEL:-model_bs_roformer_ep_317_sdr_12.9755.ckpt}"
    echo "  UVR5_DEVICE: ${UVR5_DEVICE:-auto}"
    echo "  UVR5_MAX_WORKERS: ${UVR5_MAX_WORKERS:-1}"
    
    # Check if current model exists
    local current_model_path="$uvr5_models_dir/$current_model"
    if [ -f "$current_model_path" ]; then
        echo ""
        echo "✅ 系統狀態: UVR5 已準備就緒"
    else
        echo ""
        echo "⚠️ 系統狀態: 找不到指定的模型檔案"
        echo "   缺失: $current_model_path"
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
