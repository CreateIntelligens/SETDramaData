#!/usr/bin/env python3
"""
Pyannote.audio 正規離線載入模組
使用官方推薦的離線部署方法
"""

import os
import sys
from pathlib import Path
import traceback
import warnings

# 靜音警告
warnings.filterwarnings("ignore")

class OfflinePipelineLoader:
    """正規的離線 Pipeline 載入器"""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.models_dir = self.project_root / "models"
        self.config_path = self.models_dir / "config.yaml"
        
        # 設定離線環境變數
        self._set_offline_environment()
    
    def _set_offline_environment(self):
        """設定完全離線環境變數"""
        offline_vars = {
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1',
            'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
            'HF_DATASETS_OFFLINE': '1',
            'HF_HUB_DISABLE_PROGRESS_BARS': '1',
            'TRANSFORMERS_VERBOSITY': 'error',
            'TOKENIZERS_PARALLELISM': 'false'
        }
        
        for key, value in offline_vars.items():
            os.environ[key] = value
    
    def verify_model_files(self):
        """驗證所需的模型檔案是否存在"""
        required_files = {
            "pyannote_model_segmentation-3.0.bin": "分割模型",
            "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin": "嵌入模型",
            "config.yaml": "配置檔案"
        }
        
        missing_files = []
        for filename, description in required_files.items():
            filepath = self.models_dir / filename
            if not filepath.exists():
                missing_files.append((filename, description))
        
        return missing_files
    
    def load_pipeline(self):
        """
        載入離線 Pipeline
        支援兩種載入方式：新的正規方式和舊的 HuggingFace 方式
        """
        # 先嘗試新的正規方式
        missing_files = self.verify_model_files()
        if not missing_files:
            print("🎯 嘗試新的正規離線方式...")
            try:
                return self._load_pipeline_new_method()
            except Exception as e:
                print(f"⚠️ 新方式失敗: {e}")
                print("🔄 回退到舊版本載入方式...")
        
        # 回退到舊版本 HuggingFace 方式
        return self._load_pipeline_legacy_method()
    
    def _load_pipeline_new_method(self):
        """新的正規離線方式（使用 .bin 檔案）"""
        original_cwd = Path.cwd().resolve()
        
        try:
            os.chdir(self.models_dir)
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(str(self.config_path))
            return pipeline
        finally:
            os.chdir(original_cwd)
    
    def _load_pipeline_legacy_method(self):
        """舊版本 HuggingFace 載入方式"""
        print("🔄 使用舊版本 HuggingFace 載入方式...")
        
        # 查找 HuggingFace 目錄中的配置檔案
        hf_dir = self.models_dir / "huggingface" / "models--pyannote--speaker-diarization-3.1"
        
        if not hf_dir.exists():
            raise FileNotFoundError(f"找不到 HuggingFace 模型目錄: {hf_dir}")
        
        # 查找快照目錄中的 config.yaml
        snapshots_dir = hf_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot_dir in snapshots_dir.iterdir():
                config_file = snapshot_dir / "config.yaml"
                if config_file.exists():
                    print(f"📁 找到配置檔案: {config_file}")
                    
                    from pyannote.audio import Pipeline
                    pipeline = Pipeline.from_pretrained(str(config_file))
                    print("✅ 舊版本載入方式成功！")
                    return pipeline
        
        raise FileNotFoundError("找不到有效的 HuggingFace 配置檔案")
    
    def setup_gpu_if_available(self, pipeline):
        """如果有 GPU 可用，將 Pipeline 移到 GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                pipeline.to(device)
                return device.type
            else:
                return "cpu"
        except Exception:
            return "cpu"

def load_offline_pipeline(project_root=None):
    """
    便利函數：載入離線 Pipeline
    
    Args:
        project_root: 專案根目錄路徑
        
    Returns:
        tuple: (pipeline, device_type)
    """
    loader = OfflinePipelineLoader(project_root)
    pipeline = loader.load_pipeline()
    device_type = loader.setup_gpu_if_available(pipeline)
    
    return pipeline, device_type

def test_offline_pipeline(project_root=None, force_legacy=False):
    """測試離線 Pipeline 載入"""
    print("🎯 測試正規離線 Pipeline 載入")
    print("=" * 50)
    
    try:
        loader = OfflinePipelineLoader(project_root)
        
        if force_legacy:
            print("🔄 強制使用舊版本載入方式...")
            pipeline = loader._load_pipeline_legacy_method()
        else:
            # 驗證檔案
            print("📁 驗證模型檔案...")
            missing_files = loader.verify_model_files()
            if missing_files:
                print("❌ 缺少檔案:")
                for filename, desc in missing_files:
                    print(f"  - {desc}: {filename}")
                return False
            print("✅ 所有檔案存在")
            
            # 載入 Pipeline
            print("🚀 載入 Pipeline...")
            pipeline = loader.load_pipeline()
        
        print("✅ Pipeline 載入成功")
        
        # 設定 GPU
        device_type = loader.setup_gpu_if_available(pipeline)
        print(f"🎮 使用設備: {device_type.upper()}")
        
        # 基本資訊
        print(f"📊 Pipeline 類型: {type(pipeline).__name__}")
        
        print("🎉 測試完成 - 離線 Pipeline 工作正常！")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_offline_pipeline()
    sys.exit(0 if success else 1)