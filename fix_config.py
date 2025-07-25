#!/usr/bin/env python3
"""
修復 H100 環境中損壞的 config.yaml 檔案
"""

import os
from pathlib import Path

def fix_diarization_config():
    """修復 speaker-diarization-3.1 的配置檔案"""
    print("🔧 修復 speaker-diarization-3.1 配置檔案...")
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models" / "huggingface"
    
    config_path = models_dir / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "84fd25912480287da0247647c3d2b4853cb3ee5d" / "config.yaml"
    
    # 正確的配置內容
    config_content = """task:
  _target_: pyannote.audio.tasks.SpeakerDiarization
  duration: 10.0
  max_speakers_per_chunk: 3
  max_speakers_per_frame: 2

version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    segmentation:
      model: pyannote/segmentation-3.0
    embedding:
      model: pyannote/wespeaker-voxceleb-resnet34-LM
    clustering:
      method: centroid
      min_cluster_size: 12
      threshold: 0.7045654963945799
    segmentation_step: 0.1
    embedding_step: 0.1
    embedding_batch_size: 1
    segmentation_batch_size: 1
"""
    
    print(f"📁 配置檔案路徑: {config_path}")
    
    try:
        # 備份原始檔案（如果存在且不為空）
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                original_content = f.read().strip()
            
            if original_content:
                backup_path = config_path.with_suffix('.yaml.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"📋 已備份原始配置到: {backup_path}")
        
        # 寫入正確的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print("✅ 配置檔案修復成功")
        
        # 驗證修復結果
        with open(config_path, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        if new_content.strip():
            print("✅ 配置檔案驗證成功")
            return True
        else:
            print("❌ 配置檔案仍然為空")
            return False
            
    except Exception as e:
        print(f"❌ 修復失敗: {e}")
        return False

def main():
    print("🔧 H100 配置檔案修復工具")
    print("=" * 50)
    
    success = fix_diarization_config()
    
    if success:
        print("\n✅ 修復完成！現在可以重新測試 diarization pipeline")
        print("💡 執行: python test_offline_complete.py")
    else:
        print("\n❌ 修復失敗，請檢查檔案權限")

if __name__ == "__main__":
    main()
