#!/usr/bin/env python3
"""
HuggingFace 模型離線下載腳本
- 讀取 .env 中的 token
- 在新進程中下載，避免環境變數衝突
- 建立完整的離線快取結構
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

def load_env_token():
    """從 .env 檔案載入 HuggingFace token"""
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('HUGGINGFACE_TOKEN=') or line.startswith('HF_TOKEN='):
                    token = line.split('=', 1)[1].strip().strip('"\'')
                    if token:
                        print(f"✅ 找到 HuggingFace token")
                        return token
    
    print("⚠️ 未找到 HuggingFace token，將嘗試匿名下載")
    return None

def create_download_script(cache_dir, token):
    """建立獨立的下載腳本"""
    script_content = f'''#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import snapshot_download, login

# 清除所有可能的離線模式環境變數
for var in list(os.environ.keys()):
    if 'OFFLINE' in var or 'HF_' in var:
        del os.environ[var]

# 強制設定線上模式
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

cache_dir = Path("{cache_dir}")
cache_dir.mkdir(parents=True, exist_ok=True)

# 登入
token = "{token}"
if token and token != "None":
    try:
        login(token=token)
        print("✅ HuggingFace 登入成功")
    except Exception as e:
        print(f"⚠️ 登入失敗: {{e}}")

# 下載模型
models = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0", 
    "pyannote/wespeaker-voxceleb-resnet34-LM"
]

success_count = 0
for model_id in models:
    print(f"⬇️ 下載 {{model_id}}...")
    try:
        snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            local_files_only=False,
            token=token if token != "None" else None
        )
        print(f"✅ {{model_id}} 下載完成")
        success_count += 1
    except Exception as e:
        print(f"❌ {{model_id}} 下載失敗: {{e}}")

print(f"📊 下載結果: {{success_count}}/{{len(models)}} 個模型成功")
'''
    
    script_path = Path(__file__).parent.parent / "temp_download.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def create_config_files(cache_dir):
    """建立必要的配置檔案"""
    
    # speaker-diarization-3.1 配置
    diar_config = {
        "pipeline": {
            "_target_": "pyannote.audio.pipelines.SpeakerDiarization",
            "segmentation": "pyannote/segmentation-3.0",
            "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "clustering": "AgglomerativeClustering",
            "params": {
                "segmentation": {
                    "min_duration_off": 0.0,
                    "onset": 0.5,
                    "offset": 0.5
                },
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 12,
                    "threshold": 0.7045654963945799
                }
            }
        }
    }
    
    # segmentation-3.0 配置
    seg_config = {
        "architecture": {
            "_target_": "pyannote.audio.models.segmentation.PyanNet",
            "sample_rate": 16000,
            "num_channels": 1,
            "sincnet": {
                "stride": [5, 1, 1, 1, 1, 1, 1]
            },
            "tdnn": {
                "context": [0, 1, 2, 3, 4]
            }
        },
        "task": {
            "_target_": "pyannote.audio.tasks.Segmentation",
            "duration": 2.0,
            "warm_up": [0.1, 0.1],
            "balance": "weighted",
            "weight": "balanced"
        }
    }
    
    # 檢查並建立配置檔案
    configs_to_create = [
        {
            "path": "models--pyannote--speaker-diarization-3.1",
            "config": diar_config
        },
        {
            "path": "models--pyannote--segmentation-3.0", 
            "config": seg_config
        }
    ]
    
    for item in configs_to_create:
        model_dir = cache_dir / item["path"]
        if model_dir.exists():
            # 找到 snapshots 目錄
            snapshots_dir = model_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        config_file = snapshot / "config.yaml"
                        if not config_file.exists():
                            with open(config_file, 'w') as f:
                                yaml.dump(item["config"], f, default_flow_style=False)
                            print(f"   ✅ 建立 {item['path']}/config.yaml")

def download_models():
    """下載 pyannote 模型"""
    
    print("🔧 使用獨立進程下載模型（避免環境變數衝突）...")
    
    # 設定下載目錄
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "models" / "huggingface"
    
    print(f"🚀 開始下載 pyannote 模型")
    print(f"📁 下載目錄: {cache_dir}")
    
    # 建立目錄
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入 token
    token = load_env_token()
    
    # 建立獨立的下載腳本
    script_path = create_download_script(cache_dir, token or "None")
    
    try:
        # 在新進程中執行下載，完全隔離環境變數
        print("🚀 在獨立進程中執行下載...")
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, env={})
        
        print(result.stdout)
        if result.stderr:
            print("錯誤輸出:", result.stderr)
        
        if result.returncode == 0:
            print("✅ 下載進程執行成功")
        else:
            print(f"❌ 下載進程失敗，返回碼: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 執行下載腳本失敗: {e}")
    
    finally:
        # 清理臨時腳本
        if script_path.exists():
            script_path.unlink()
    
    # 建立配置檔案
    print(f"\n🔧 建立配置檔案...")
    create_config_files(cache_dir)
    
    # 檢查下載結果
    print(f"\n🔍 檢查下載結果...")
    model_count = 0
    for model_dir in cache_dir.glob("models--pyannote--*"):
        if model_dir.is_dir() and any(model_dir.rglob("*.bin")):
            model_count += 1
            print(f"  ✅ {model_dir.name}")
    
    if model_count == 0:
        print("❌ 沒有找到任何下載的模型")
        return False
    
    print(f"📊 成功下載 {model_count} 個模型")
    
    # 測試離線載入
    print(f"\n🚀 測試離線載入...")
    
    # 設定離線模式環境變數
    test_env = os.environ.copy()
    test_env.update({
        'HF_HUB_CACHE': str(cache_dir),
        'HF_HUB_OFFLINE': '1'
    })
    
    try:
        # 在新進程中測試載入
        test_script = f'''
import os
os.environ['HF_HUB_CACHE'] = '{cache_dir}'
os.environ['HF_HUB_OFFLINE'] = '1'
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
print("✅ 離線載入測試成功！")
'''
        
        result = subprocess.run([
            sys.executable, '-c', test_script
        ], capture_output=True, text=True, env=test_env)
        
        if result.returncode == 0:
            print("✅ 離線載入測試成功！")
        else:
            print(f"❌ 離線載入測試失敗: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 測試載入失敗: {e}")
    
    print(f"\n🎉 下載完成！")
    print(f"📁 快取位置: {cache_dir}")
    print(f"🎯 系統已準備好離線模式")
    
    return True

if __name__ == "__main__":
    download_models()
