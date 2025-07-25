#!/usr/bin/env python3
"""
除錯 Diarization Pipeline 載入問題
"""

import os
import sys
from pathlib import Path
import yaml

# 設定環境變數
os.environ.update({
    'TRANSFORMERS_OFFLINE': '1',
    'HUGGINGFACE_HUB_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1'
})

project_root = Path(__file__).parent
models_dir = project_root / "models" / "huggingface"

def check_diarization_config():
    """檢查 diarization 配置檔案"""
    print("🔍 檢查 Diarization 配置檔案...")
    
    # 檢查配置檔案路徑
    config_path = models_dir / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "84fd25912480287da0247647c3d2b4853cb3ee5d" / "config.yaml"
    
    print(f"📁 配置檔案路徑: {config_path}")
    print(f"📁 檔案存在: {config_path.exists()}")
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            print(f"📄 配置檔案內容:")
            print("=" * 50)
            print(config_content)
            print("=" * 50)
            
            # 嘗試解析 YAML
            try:
                config_data = yaml.safe_load(config_content)
                print(f"✅ YAML 解析成功")
                print(f"📊 配置結構: {list(config_data.keys()) if config_data else 'None'}")
                
                # 檢查 pipeline 配置
                if 'pipeline' in config_data:
                    pipeline_config = config_data['pipeline']
                    print(f"🔧 Pipeline 配置: {pipeline_config}")
                    
                    if 'params' in pipeline_config:
                        params = pipeline_config['params']
                        print(f"⚙️ Pipeline 參數: {params}")
                        
                        # 檢查是否有 None 值
                        for key, value in params.items() if params else []:
                            if value is None:
                                print(f"⚠️ 發現 None 值: {key} = {value}")
                
            except yaml.YAMLError as e:
                print(f"❌ YAML 解析失敗: {e}")
                
        except Exception as e:
            print(f"❌ 讀取配置檔案失敗: {e}")
    
    # 檢查相關模型檔案
    print(f"\n🔍 檢查相關模型檔案...")
    snapshot_dir = models_dir / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "84fd25912480287da0247647c3d2b4853cb3ee5d"
    
    if snapshot_dir.exists():
        files = list(snapshot_dir.iterdir())
        print(f"📁 Snapshot 目錄檔案:")
        for file in files:
            print(f"  - {file.name}")
    
    # 檢查 segmentation 和 embedding 模型路徑
    print(f"\n🔍 檢查子模型路徑...")
    
    # segmentation-3.0
    seg_path = models_dir / "models--pyannote--segmentation-3.0" / "snapshots" / "e66f3d3b9eb0873085418a7b813d3b369bf160bb"
    print(f"📁 Segmentation 路徑存在: {seg_path.exists()}")
    
    # embedding
    emb_path = models_dir / "models--pyannote--wespeaker-voxceleb-resnet34-LM" / "snapshots" / "837717ddb9ff5507820346191109dc79c958d614"
    print(f"📁 Embedding 路徑存在: {emb_path.exists()}")

def test_manual_load():
    """嘗試手動載入 pipeline"""
    print(f"\n🧪 嘗試手動載入 Pipeline...")
    
    try:
        import torch
        from pyannote.audio import Pipeline
        
        print(f"📦 PyTorch 版本: {torch.__version__}")
        
        # 嘗試載入
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=str(models_dir),
            use_auth_token=None
        )
        print(f"✅ Pipeline 載入成功!")
        
    except Exception as e:
        print(f"❌ Pipeline 載入失敗: {e}")
        print(f"🔍 錯誤類型: {type(e).__name__}")
        
        # 詳細錯誤資訊
        import traceback
        print(f"📋 完整錯誤:")
        traceback.print_exc()

def main():
    print("🔍 Diarization Pipeline 除錯工具")
    print("=" * 50)
    
    check_diarization_config()
    test_manual_load()

if __name__ == "__main__":
    main()
