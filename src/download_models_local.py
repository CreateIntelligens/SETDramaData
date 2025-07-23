#!/usr/bin/env python3
"""
SETVoicePrep - 模型下載工具
下載pyannote模型到專案內的models目錄，用於離線部署

使用方法:
    python download_models_local.py
    
注意:
    1. 需要設定.env檔案中的HUGGINGFACE_TOKEN
    2. 需要先到以下頁面授權存取：
       - https://huggingface.co/pyannote/embedding
"""

import os
import sys
from pathlib import Path

def load_env():
    """載入.env檔案"""
    env_file = Path(__file__).parent.parent / ".env"  # 往上一層到專案根目錄
    if env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print(f"📋 載入環境變數: {env_file}")
        except UnicodeDecodeError:
            # 嘗試其他編碼
            try:
                with open(env_file, 'r', encoding='cp950') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                print(f"📋 載入環境變數: {env_file} (cp950編碼)")
            except Exception as e:
                print(f"❌ 無法讀取.env檔案: {e}")
    else:
        print(f"⚠️ 找不到.env檔案: {env_file}")

def download_to_local():
    """下載模型到專案內的models目錄"""
    
    # 檢查HF token
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ 錯誤: 需要設定HuggingFace token")
        print("   請在.env檔案中設定 HUGGINGFACE_TOKEN=your_token")
        return False
    
    # 建立models目錄
    models_dir = Path(__file__).parent.parent / "models"  # 往上一層到專案根目錄
    models_dir.mkdir(exist_ok=True)
    
    # 建立子目錄
    (models_dir / "huggingface").mkdir(exist_ok=True)
    (models_dir / "torch").mkdir(exist_ok=True)
    
    print(f"📁 模型將下載到: {models_dir}")
    
    # 設定HuggingFace cache目錄到專案內
    os.environ['HF_HOME'] = str(models_dir / "huggingface")
    os.environ['TORCH_HOME'] = str(models_dir / "torch")
    os.environ['TRANSFORMERS_CACHE'] = str(models_dir / "huggingface")
    os.environ['HF_HUB_CACHE'] = str(models_dir / "huggingface" / "hub")
    
    print(f"🔧 設定緩存目錄:")
    print(f"   HF_HOME: {os.environ['HF_HOME']}")
    print(f"   TORCH_HOME: {os.environ['TORCH_HOME']}")
    
    print("📥 開始下載模型...")
    
    # 下載diarization模型
    try:
        from pyannote.audio import Pipeline
        print("1. 下載 diarization 模型...")
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', 
                                          use_auth_token=hf_token, 
                                          cache_dir=str(models_dir / "huggingface"))
        print("✅ Diarization 模型下載完成")
    except Exception as e:
        print(f"❌ Diarization 模型下載失敗: {e}")
        return False
    
    # 下載embedding模型
    try:
        from pyannote.audio import Model
        print("2. 下載 embedding 模型...")
        model = Model.from_pretrained('pyannote/embedding', 
                                    use_auth_token=hf_token,
                                    cache_dir=str(models_dir / "huggingface"))
        print("✅ Embedding 模型下載完成")
    except Exception as e:
        print(f"❌ Embedding 模型下載失敗: {e}")
        print(f"   請確認已授權存取 https://huggingface.co/pyannote/embedding")
        return False
    
    
    # 檢查下載結果
    print(f"\n📊 檢查下載結果:")
    hf_dir = models_dir / "huggingface"
    if hf_dir.exists() and list(hf_dir.glob("**/*")):
        print(f"✅ HuggingFace 模型: {len(list(hf_dir.glob('**/*')))} 個檔案")
    else:
        print("⚠️ HuggingFace 模型目錄為空")
    
    print(f"\n🎉 模型已下載到: {models_dir}")
    print(f"📦 打包時請包含整個 models/ 目錄")
    
    return True

if __name__ == "__main__":
    load_env()
    download_to_local()
