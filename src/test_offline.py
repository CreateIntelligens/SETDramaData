#!/usr/bin/env python3
"""
完全離線模型測試腳本
"""
import os
from pathlib import Path

    # 🔥 強制離線模式
os.environ.update({
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1'
    })

    # 設定本地模型路徑
project_root = Path('/workspace/etl')  # 或你的專案路徑
models_dir = project_root / "models"

if models_dir.exists():
    os.environ.update({
        'HF_HOME': str(models_dir / "huggingface"),
        'TORCH_HOME': str(models_dir / "torch"),
        'HF_HUB_CACHE': str(models_dir / "huggingface" / "hub"),
    })
    print(f"🔧 使用本地模型: {models_dir}")
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
        local_diar_path = models_dir / "huggingface" /
    "models--pyannote--speaker-diarization-3.1"
     print(f"   📁 檢查路徑: {local_diar_path}")
      print(f"   📁 路徑存在: {local_diar_path.exists()}")

       if local_diar_path.exists():
            pipeline = Pipeline.from_pretrained(str(local_diar_path))
            print("   ✅ Diarization 模型載入成功（本地）")
        else:
            print("   ❌ 本地模型路徑不存在")
            # 列出實際存在的目錄
            if (models_dir / "huggingface").exists():
                print("   📋 實際存在的模型目錄:")
                for p in (models_dir / "huggingface").iterdir():
                    if p.is_dir():
                        print(f"      {p.name}")

        print("\n🔊 測試 Embedding 模型載入...")
        local_emb_path = models_dir / "huggingface" /
    "models--pyannote--embedding"
     print(f"   📁 檢查路徑: {local_emb_path}")
      print(f"   📁 路徑存在: {local_emb_path.exists()}")

       if local_emb_path.exists():
            model = Model.from_pretrained(str(local_emb_path))
            print("   ✅ Embedding 模型載入成功（本地）")
        else:
            print("   ❌ 本地模型路徑不存在")

        print("\n🎉 所有模型測試通過 - 完全離線運作!")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
