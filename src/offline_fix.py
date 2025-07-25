#!/usr/bin/env python3
"""
離線環境修復腳本
解決 Transformers 快取版本問題
"""

import os
import shutil
from pathlib import Path

def fix_offline_cache():
    """修復離線快取問題"""
    print("🔧 修復離線環境設定...")
    
    # 找到專案根目錄
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    
    # 設定環境變數
    env_vars = {
        'HF_HOME': str(models_dir / "huggingface"),
        'TORCH_HOME': str(models_dir / "torch"),
        'HF_HUB_CACHE': str(models_dir / "huggingface" / "hub"),
        'TRANSFORMERS_CACHE': str(models_dir / "huggingface"),
        'TRANSFORMERS_OFFLINE': '0',  # 暫時設為線上模式
        'HF_DATASETS_OFFLINE': '0',
        'HF_HUB_OFFLINE': '0'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key} = {value}")
    
    print("✅ 環境變數設定完成")
    
    # 清理可能有問題的快取
    cache_dirs = [
        models_dir / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch"
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"🧹 清理快取目錄: {cache_dir}")
            try:
                # 只清理 .lock 檔案
                for lock_file in cache_dir.rglob("*.lock"):
                    lock_file.unlink()
                    print(f"   刪除 lock 檔案: {lock_file}")
            except Exception as e:
                print(f"   ⚠️ 清理失敗: {e}")

if __name__ == "__main__":
    fix_offline_cache()
    print("\n✅ 離線環境修復完成！")