#!/usr/bin/env python3
"""
etl - 模型下載工具
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
    
    # 建立models目錄 - 確保在專案根目錄
    current_dir = Path.cwd()
    
    # 如果當前目錄是 src，往上一層
    if current_dir.name == 'src':
        project_root = current_dir.parent
    # 如果當前目錄包含 src 目錄，就是專案根目錄
    elif (current_dir / 'src').exists():
        project_root = current_dir
    # 否則使用腳本所在目錄的上一層
    else:
        project_root = Path(__file__).parent.parent
    
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"📍 專案根目錄: {project_root}")
    print(f"📁 模型目錄: {models_dir}")
    
    # 確保目錄正確
    if not (project_root / 'src').exists():
        print(f"⚠️ 警告: 在 {project_root} 找不到 src 目錄，請確認路徑正確")
        response = input("是否繼續？(y/N): ")
        if response.lower() != 'y':
            return False
    
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
    
    # 下載所有相關模型（包含依賴）
    models_to_download = [
        ("pyannote/speaker-diarization-3.1", "speaker-diarization-3.1"),
        ("pyannote/segmentation-3.0", "segmentation-3.0"),
        ("pyannote/embedding", "embedding"),
        ("pyannote/wespeaker-voxceleb-resnet34-LM", "wespeaker-voxceleb-resnet34-LM"),
    ]
    
    import huggingface_hub
    
    for i, (repo_id, dir_name) in enumerate(models_to_download, 1):
        try:
            print(f"{i}. 下載 {repo_id}...")
            
            # 直接下載到指定目錄，不使用快取
            local_dir = models_dir / "direct" / dir_name
            local_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"   📥 直接下載到: {local_dir}")
            
            # 使用 snapshot_download 直接下載所有檔案
            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                token=hf_token,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False  # 🔥 關鍵：不使用符號連結
            )
            
            print(f"✅ {repo_id} 下載完成")
            
            # 檢查關鍵檔案
            config_file = local_dir / "config.yaml"
            if config_file.exists():
                print(f"   ✅ config.yaml 已下載")
            else:
                print(f"   ⚠️ config.yaml 缺失")
                
        except Exception as e:
            print(f"❌ {repo_id} 下載失敗: {e}")
            return False
    
    
    
    # 檢查下載結果
    print(f"\n📊 檢查下載結果:")
    hf_dir = models_dir / "huggingface"
    if hf_dir.exists() and list(hf_dir.glob("**/*")):
        print(f"✅ HuggingFace 模型: {len(list(hf_dir.glob('**/*')))} 個檔案")
    else:
        print("⚠️ HuggingFace 模型目錄為空")
    
    # 創建 HuggingFace 快取格式的符號連結/複製
    print(f"\n🔧 建立快取格式連結...")
    
    try:
        import shutil
        
        # 為所有模型建立快取格式
        cache_models = [
            ("speaker-diarization-3.1", "models--pyannote--speaker-diarization-3.1", "84fd25912480287da0247647c3d2b4853cb3ee5d"),
            ("segmentation-3.0", "models--pyannote--segmentation-3.0", "9ba7a982f78ed6c4bf7829c8c6b6b9b66f4e8e1a"),
            ("embedding", "models--pyannote--embedding", "4db4899737a38b2d618bbd74350915aa10293cb2"),
            ("wespeaker-voxceleb-resnet34-LM", "models--pyannote--wespeaker-voxceleb-resnet34-LM", "5c8f693e982832d4a0c07323c69b8cdcfaacbc70"),
        ]
        
        for direct_name, cache_name, hash_id in cache_models:
            # 創建快取目錄結構
            cache_dir = models_dir / "huggingface" / cache_name / "snapshots"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_dir = cache_dir / hash_id
            
            if not snapshot_dir.exists():
                source_dir = models_dir / "direct" / direct_name
                if source_dir.exists():
                    shutil.copytree(source_dir, snapshot_dir)
                    print(f"   ✅ 建立 {cache_name} 快取格式")
            
            # 建立 refs/main
            refs_dir = models_dir / "huggingface" / cache_name / "refs"
            refs_dir.mkdir(exist_ok=True)
            with open(refs_dir / "main", 'w') as f:
                f.write(hash_id)
        
        print("✅ 所有快取格式建立完成")
        
    except Exception as e:
        print(f"⚠️ 快取格式建立失敗: {e}")
        print("   直接下載的模型仍可使用")
    
    # 修改 config.yaml 中的路徑引用為本地路徑
    print(f"\n🔧 修改配置檔案路徑引用...")
    try:
        diar_config = models_dir / "direct" / "speaker-diarization-3.1" / "config.yaml"
        if diar_config.exists():
            # 讀取配置
            with open(diar_config, 'r') as f:
                content = f.read()
            
            # 替換 repo ID 為本地路徑
            content = content.replace(
                'embedding: pyannote/wespeaker-voxceleb-resnet34-LM',
                f'embedding: {models_dir / "direct" / "wespeaker-voxceleb-resnet34-LM"}'
            )
            content = content.replace(
                'segmentation: pyannote/segmentation-3.0',
                f'segmentation: {models_dir / "direct" / "segmentation-3.0"}'
            )
            
            # 寫回檔案
            with open(diar_config, 'w') as f:
                f.write(content)
            
            print("✅ 配置檔案路徑已修改為本地路徑")
        else:
            print("⚠️ 找不到 diarization config.yaml")
            
    except Exception as e:
        print(f"⚠️ 配置檔案修改失敗: {e}")
    
    print(f"\n🎉 模型已下載到: {models_dir}")
    print(f"📦 打包時請包含整個 models/ 目錄")
    
    return True

if __name__ == "__main__":
    load_env()
    download_to_local()
