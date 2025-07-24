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