#!/usr/bin/env python3
"""
設定 HuggingFace 快取的索引檔案
"""
import os
import json
from pathlib import Path

# 設定路徑
models_dir = Path("models/huggingface")

# 模型配置
models_config = {
    "models--pyannote--speaker-diarization-3.1": {
        "snapshot_id": "84fd25912480287da0247647c3d2b4853cb3ee5d",
        "repo_id": "pyannote/speaker-diarization-3.1"
    },
    "models--pyannote--embedding": {
        "snapshot_id": "4db4899737a38b2d618bbd74350915aa10293cb2", 
        "repo_id": "pyannote/embedding"
    }
}

def setup_cache_files():
    """建立必要的快取索引檔案"""
    
    for model_name, config in models_config.items():
        model_dir = models_dir / model_name
        snapshot_id = config["snapshot_id"]
        repo_id = config["repo_id"]
        
        print(f"🔧 設定 {model_name}")
        
        # 確保目錄存在
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立 .huggingface 目錄
        hf_dir = model_dir / ".huggingface"
        hf_dir.mkdir(exist_ok=True)
        
        # 建立 download_metadata.json
        metadata = {
            "url": f"https://huggingface.co/{repo_id}",
            "etag": snapshot_id,
            "repo_id": repo_id,
            "snapshot_id": snapshot_id
        }
        
        metadata_file = hf_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ 建立 {metadata_file}")
        
        # 確保 refs/main 指向正確的 snapshot
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(exist_ok=True)
        
        main_ref = refs_dir / "main"
        with open(main_ref, 'w') as f:
            f.write(snapshot_id)
        
        print(f"   ✅ 更新 {main_ref}")
        
        # 檢查 snapshot 目錄是否存在
        snapshot_dir = model_dir / "snapshots" / snapshot_id
        if snapshot_dir.exists():
            print(f"   ✅ Snapshot 目錄存在: {snapshot_dir}")
        else:
            print(f"   ❌ Snapshot 目錄不存在: {snapshot_dir}")

if __name__ == "__main__":
    print("🚀 設定 HuggingFace 快取索引檔案...")
    setup_cache_files()
    print("\n🎉 快取設定完成！")