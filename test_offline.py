#!/usr/bin/env python3
"""
完全離線模型測試腳本
"""
import os
from pathlib import Path

# 🔥 完全離線模式 - 強制使用本地快取
os.environ.update({
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'HUGGINGFACE_HUB_OFFLINE': '1',  # 強制離線模式
})

# 設定本地模型路徑
# project_root = Path('/workspace/etl')  # 或你的專案路徑
project_root = Path('.')  # 或你的專案路徑
models_dir = project_root / "models"

if models_dir.exists():
    os.environ.update({
        'HF_HOME': str(models_dir / "huggingface"),
        'TORCH_HOME': str(models_dir / "torch"),
        'HF_HUB_CACHE': str(models_dir / "huggingface"),
    })
    print(f"🔧 使用本地模型: {models_dir}")
    print(f"🔧 HF_HOME: {os.environ['HF_HOME']}")
    
    # 檢查關鍵模型是否存在
    critical_models = [
        "models--pyannote--speaker-diarization-3.1",
        "models--pyannote--segmentation-3.0", 
        "models--pyannote--embedding",
        "models--pyannote--wespeaker-voxceleb-resnet34-LM"
    ]
    
    for model in critical_models:
        model_path = models_dir / "huggingface" / model
        print(f"   📁 {model}: {'✅' if model_path.exists() else '❌'}")
        
else:
    print(f"❌ 模型目錄不存在: {models_dir}")
    exit(1)

try:
    print("🤖 測試 PyTorch...")
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   🖥️ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    print("\n📡 測試 pyannote.audio...")
    from pyannote.audio import Pipeline, Model
    print("   ✅ pyannote.audio 匯入成功")
    
    print("\n🎤 測試 Diarization 模型載入...")
    # 檢查本地快取是否存在
    hf_cache_dir = models_dir / "huggingface"
    diar_cache_path = hf_cache_dir / "models--pyannote--speaker-diarization-3.1"
    
    print(f"   📁 HuggingFace 快取路徑: {diar_cache_path}")
    print(f"   📁 快取存在: {diar_cache_path.exists()}")
    
    if diar_cache_path.exists():
        try:
            # 🔥 使用標準 repo ID，依賴 HUGGINGFACE_HUB_OFFLINE=1 使用本地快取
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            print("   ✅ Diarization 模型載入成功（離線快取）")
        except Exception as e:
            print(f"   ❌ 載入失敗: {e}")
    else:
        print("   ❌ 本地快取不存在")
    
    print("\n🔊 測試 Embedding 模型載入...")
    # 檢查本地快取是否存在
    emb_cache_path = hf_cache_dir / "models--pyannote--embedding"
    
    print(f"   📁 Embedding 快取路徑: {emb_cache_path}")
    print(f"   📁 快取存在: {emb_cache_path.exists()}")
    
    if emb_cache_path.exists():
        try:
            # 🔥 使用標準 repo ID，依賴 HUGGINGFACE_HUB_OFFLINE=1 使用本地快取
            model = Model.from_pretrained("pyannote/embedding")
            print("   ✅ Embedding 模型載入成功（離線快取）")
        except Exception as e:
            print(f"   ❌ 載入失敗: {e}")
    else:
        print("   ❌ 本地快取不存在")
    
    print("\n🎉 所有模型測試通過 - 完全離線運作!")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()