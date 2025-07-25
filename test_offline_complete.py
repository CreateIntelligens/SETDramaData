#!/usr/bin/env python3
"""
完全離線模式測試腳本
測試使用絕對路徑載入 pyannote 模型
"""

import os
import sys
from pathlib import Path

# 🎯 完全離線設定 - 使用絕對路徑，避開 HuggingFace Hub
project_root = Path(__file__).parent
models_dir = project_root / "models" / "huggingface"

# 強制離線模式環境變數
os.environ.update({
    'TRANSFORMERS_OFFLINE': '1',
    'HUGGINGFACE_HUB_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1'
})

print(f"🔧 模型目錄: {models_dir}")
print(f"🔧 完全離線模式: 已啟用")

import torch
from pyannote.audio import Pipeline, Model

def fix_huggingface_symlinks():
    """自動修復 HuggingFace 快取中損壞的符號連結"""
    print("🔧 檢查並修復 HuggingFace 符號連結...")
    
    # 定義需要修復的符號連結
    symlink_fixes = [
        {
            "model": "segmentation-3.0",
            "snapshot": "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
            "file": "pytorch_model.bin",
            "blob": "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
        },
        {
            "model": "wespeaker-voxceleb-resnet34-LM", 
            "snapshot": "837717ddb9ff5507820346191109dc79c958d614",
            "file": "pytorch_model.bin",
            "blob": "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
        },
        {
            "model": "speaker-diarization-3.1",
            "snapshot": "84fd25912480287da0247647c3d2b4853cb3ee5d",
            "file": "config.yaml",
            "blob": "5402e3ca79b6cfa5b0ec283eed920cafe45ee39b"
        }
    ]
    
    fixed_count = 0
    
    for fix in symlink_fixes:
        model_name = fix["model"]
        snapshot_id = fix["snapshot"]
        file_name = fix["file"]
        blob_id = fix["blob"]
        
        # 構建路徑
        snapshot_dir = models_dir / f"models--pyannote--{model_name}" / "snapshots" / snapshot_id
        target_file_path = snapshot_dir / file_name
        blob_path = models_dir / f"models--pyannote--{model_name}" / "blobs" / blob_id
        
        # 檢查是否需要修復
        needs_fix = False
        
        if not target_file_path.exists():
            needs_fix = True
        elif target_file_path.is_file() and target_file_path.stat().st_size == 0:
            needs_fix = True
        elif target_file_path.is_file() and not target_file_path.is_symlink():
            needs_fix = True
        elif target_file_path.is_symlink():
            try:
                target = target_file_path.readlink()
                expected_target = Path("../../blobs") / blob_id
                if target != expected_target:
                    needs_fix = True
            except Exception:
                needs_fix = True
        
        if needs_fix:
            try:
                # 檢查 blob 檔案是否存在
                if not blob_path.exists():
                    print(f"⚠️ {model_name}/{file_name}: blob 檔案不存在: {blob_path}")
                    continue
                
                # 刪除現有檔案
                if target_file_path.exists():
                    target_file_path.unlink()
                
                # 建立符號連結
                relative_blob_path = Path("../../blobs") / blob_id
                target_file_path.symlink_to(relative_blob_path)
                
                print(f"✅ {model_name}/{file_name}: 符號連結已修復")
                fixed_count += 1
                
            except Exception as e:
                print(f"❌ {model_name}/{file_name}: 修復失敗: {e}")
                continue
    
    return fixed_count

def test_diarization_pipeline():
    """測試 diarization pipeline 載入"""
    print("\n1. 測試 Diarization Pipeline...")
    
    try:
        # Pipeline 不支援 local_files_only，只使用 cache_dir
        print(f"📁 快取目錄: {models_dir}")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=str(models_dir),
            use_auth_token=None
        )
        print("✅ Diarization pipeline 載入成功")
        return True
        
    except Exception as e:
        print(f"❌ Diarization pipeline 載入失敗: {e}")
        return False

def test_embedding_model():
    """測試 embedding 模型載入"""
    print("\n2. 測試 Embedding Model...")
    
    try:
        # 使用 repo ID + cache_dir + local_files_only 的方式
        print(f"📁 快取目錄: {models_dir}")
        model = Model.from_pretrained(
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            cache_dir=str(models_dir),
            local_files_only=True,
            use_auth_token=None
        )
        print("✅ Embedding 模型載入成功")
        return True
        
    except Exception as e:
        print(f"❌ Embedding 模型載入失敗: {e}")
        return False

def test_segmentation_model():
    """測試 segmentation 模型載入"""
    print("\n3. 測試 Segmentation Model...")
    
    try:
        # 使用 repo ID + cache_dir + local_files_only 的方式
        print(f"📁 快取目錄: {models_dir}")
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            cache_dir=str(models_dir),
            local_files_only=True,
            use_auth_token=None
        )
        print("✅ Segmentation 模型載入成功")
        return True
        
    except Exception as e:
        print(f"❌ Segmentation 模型載入失敗: {e}")
        return False

def main():
    print("🧪 完全離線模式測試")
    print("=" * 50)
    
    # 檢查 GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 使用 CPU")
    
    # 檢查模型目錄
    if not models_dir.exists():
        print(f"❌ 模型目錄不存在: {models_dir}")
        sys.exit(1)
    
    print(f"✅ 模型目錄存在: {models_dir}")
    
    # 自動修復符號連結
    print("\n🔧 檢查並修復符號連結...")
    try:
        fixed_count = fix_huggingface_symlinks()
        if fixed_count > 0:
            print(f"✅ 修復了 {fixed_count} 個符號連結")
        else:
            print("✅ 符號連結檢查完成")
    except Exception as e:
        print(f"⚠️ 符號連結修復失敗: {e}")
    
    # 測試各個模型
    results = []
    results.append(test_diarization_pipeline())
    results.append(test_embedding_model())
    results.append(test_segmentation_model())
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果")
    print("=" * 50)
    
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"🎉 所有測試通過！({success_count}/{total_count})")
        print("✅ 完全離線模式運作正常")
    else:
        print(f"⚠️ 部分測試失敗：{success_count}/{total_count}")
        print("❌ 需要檢查模型檔案或配置")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
