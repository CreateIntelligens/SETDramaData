#!/usr/bin/env python3
"""
自動修復 HuggingFace 快取中損壞的符號連結
"""

import os
from pathlib import Path

def fix_huggingface_symlinks():
    """自動修復 HuggingFace 快取中損壞的符號連結"""
    project_root = Path(__file__).parent
    models_dir = project_root / "models" / "huggingface"
    
    print("🔧 檢查並修復 HuggingFace 符號連結...")
    
    # 定義需要修復的符號連結
    symlink_fixes = [
        {
            "model": "segmentation-3.0",
            "snapshot": "e66f3d3b9eb0873085418a7b813d3b369bf160bb",
            "blob": "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
        },
        {
            "model": "wespeaker-voxceleb-resnet34-LM", 
            "snapshot": "837717ddb9ff5507820346191109dc79c958d614",
            "blob": "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"
        }
    ]
    
    fixed_count = 0
    
    for fix in symlink_fixes:
        model_name = fix["model"]
        snapshot_id = fix["snapshot"]
        blob_id = fix["blob"]
        
        # 構建路徑
        snapshot_dir = models_dir / f"models--pyannote--{model_name}" / "snapshots" / snapshot_id
        pytorch_model_path = snapshot_dir / "pytorch_model.bin"
        blob_path = models_dir / f"models--pyannote--{model_name}" / "blobs" / blob_id
        
        # 檢查是否需要修復
        needs_fix = False
        
        if not pytorch_model_path.exists():
            print(f"⚠️ {model_name}: pytorch_model.bin 不存在")
            needs_fix = True
        elif pytorch_model_path.is_file() and pytorch_model_path.stat().st_size == 0:
            print(f"⚠️ {model_name}: pytorch_model.bin 是空檔案")
            needs_fix = True
        elif pytorch_model_path.is_file() and not pytorch_model_path.is_symlink():
            print(f"⚠️ {model_name}: pytorch_model.bin 不是符號連結")
            needs_fix = True
        elif pytorch_model_path.is_symlink():
            # 檢查符號連結是否正確
            try:
                target = pytorch_model_path.readlink()
                expected_target = Path("../../blobs") / blob_id
                if target != expected_target:
                    print(f"⚠️ {model_name}: 符號連結目標錯誤 {target} != {expected_target}")
                    needs_fix = True
                else:
                    print(f"✅ {model_name}: 符號連結正常")
            except Exception as e:
                print(f"⚠️ {model_name}: 符號連結檢查失敗: {e}")
                needs_fix = True
        
        if needs_fix:
            try:
                # 檢查 blob 檔案是否存在
                if not blob_path.exists():
                    print(f"❌ {model_name}: blob 檔案不存在: {blob_path}")
                    continue
                
                # 刪除現有檔案
                if pytorch_model_path.exists():
                    pytorch_model_path.unlink()
                
                # 建立符號連結
                relative_blob_path = Path("../../blobs") / blob_id
                pytorch_model_path.symlink_to(relative_blob_path)
                
                print(f"✅ {model_name}: 符號連結已修復")
                fixed_count += 1
                
            except Exception as e:
                print(f"❌ {model_name}: 修復失敗: {e}")
    
    if fixed_count > 0:
        print(f"🎉 修復了 {fixed_count} 個符號連結")
    else:
        print("✅ 所有符號連結都正常")
    
    return fixed_count

def main():
    """主函數"""
    print("🔧 HuggingFace 符號連結修復工具")
    print("=" * 50)
    
    fixed_count = fix_huggingface_symlinks()
    
    print("\n" + "=" * 50)
    if fixed_count > 0:
        print(f"✅ 修復完成！修復了 {fixed_count} 個符號連結")
        print("💡 建議重新執行模型載入測試")
    else:
        print("✅ 所有符號連結都正常，無需修復")

if __name__ == "__main__":
    main()
