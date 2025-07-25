#!/usr/bin/env python3
"""
完全離線模型測試腳本（最終版 - 全手動）
"""
import os
from pathlib import Path
import torch
import yaml
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.embedding import WeSpeakerResNet34
from pyannote.audio.tasks import SpeakerDiarization as SpeakerDiarizationTask

class DummyProtocol:
    def __init__(self):
        self.preprocessors = {}
        self.scope = "file"  # 必需的 scope 屬性
    
    @property
    def name(self):
        return "dummy"
    
    def train(self):
        # 至少要有一個元素，避免 StopIteration
        yield {"uri": "dummy", "audio": "dummy.wav"}
    
    def development(self):
        yield {"uri": "dummy", "audio": "dummy.wav"}
    
    def test(self):
        yield {"uri": "dummy", "audio": "dummy.wav"}

def load_model_manually(model_class, config_path, checkpoint_path, device="cpu", attach_task=False):
    """
    一個輔助函數，用於手動實例化模型、載入權重，並可選擇性地附加任務。
    """
    print(f"   - 正在手動載入 {model_class.__name__}...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config.get('model', {})
    model_params.pop('_target_', None)
    
    model = model_class(**model_params)
    
    if attach_task:
        if 'task' in config:
            # 🔥 簡化但有效的 task 設定
            from pyannote.audio.core.task import Specifications
            from pyannote.core import SlidingWindow
            import torch.nn as nn
            
            # 從配置獲取參數
            task_config = config['task']
            duration = task_config.get('duration', 10.0)
            max_speakers = task_config.get('max_speakers_per_frame', 2)
            
            # 建立 specifications
            step = 0.01  # 10ms step
            window = SlidingWindow(duration=duration, step=step)
            classes = [f"speaker_{i}" for i in range(max_speakers + 1)]
            
            specifications = Specifications(
                problem="multilabel_classification",
                resolution=window,
                duration=duration,
                classes=classes,
                permutation_invariant=True
            )
            
            model._specifications = specifications
            
            # 🔥 為 PyanNet 手動建立缺失的層
            if model_class.__name__ == 'PyanNet':
                # 獲取模型維度
                linear_config = model_params.get('linear', {})
                linear_dim = linear_config.get('hidden_size', 128)
                num_classes = len(classes)
                
                # 手動建立 classifier 層
                if not hasattr(model, 'classifier'):
                    model.classifier = nn.Linear(linear_dim, num_classes)
                
                # 手動建立 activation 層
                if not hasattr(model, 'activation'):
                    model.activation = nn.Sigmoid()
                
                print(f"     🔧 已建立 classifier({linear_dim} -> {num_classes}) 和 activation 層")
    
    # PyTorch 2.6+ 預設 weights_only=True，這裡強制設為 False 以支援舊 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 🔥 為 PyanNet 根據檢查點調整 classifier 維度
    if model_class.__name__ == 'PyanNet' and attach_task:
        if 'classifier.weight' in checkpoint['state_dict']:
            # 從檢查點獲取正確的維度
            checkpoint_classifier_shape = checkpoint['state_dict']['classifier.weight'].shape
            correct_num_classes = checkpoint_classifier_shape[0]  # 輸出維度
            correct_input_dim = checkpoint_classifier_shape[1]    # 輸入維度
            
            print(f"     🔧 從檢查點檢測到 classifier 維度: {correct_input_dim} -> {correct_num_classes}")
            
            # 重新建立正確維度的 classifier
            import torch.nn as nn
            model.classifier = nn.Linear(correct_input_dim, correct_num_classes)
            
            # 更新 specifications 的類別數
            if hasattr(model, '_specifications'):
                # 建立正確數量的類別
                classes = [f"speaker_{i}" for i in range(correct_num_classes)]
                from pyannote.audio.core.task import Specifications
                from pyannote.core import SlidingWindow
                
                duration = model._specifications.duration
                window = model._specifications.resolution
                
                model._specifications = Specifications(
                    problem="multilabel_classification",
                    resolution=window,
                    duration=duration,
                    classes=classes,
                    permutation_invariant=True
                )
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(device)
    model.eval()
    print(f"     ✅ {model_class.__name__} 載入成功。")
    return model

def test_offline_diarization_fully_manual():
    """
    測試完全離線的 Speaker Diarization Pipeline 載入。
    採用最底層、最可靠的全手動載入方式。
    """
    print("="*50)
    print("🚀 開始進行 Pyannote Audio 離線載入測試 (全手動模式) 🚀")
    print("="*50)

    script_dir = Path(__file__).parent.parent
    base_model_path = script_dir / "models" / "direct"
    
    seg_dir = base_model_path / "segmentation-3.0"
    emb_dir = base_model_path / "wespeaker-voxceleb-resnet34-LM"

    print("\n🤖 1. 測試 PyTorch...")
    try:
        print(f"   ✅ PyTorch 版本: {torch.__version__}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   ✅ 使用裝置: {device.type.upper()}")
    except Exception as e:
        print(f"   ❌ PyTorch 測試失敗: {e}")
        return

    print("\n🔧 2. 手動載入子模型...")
    try:
        # 🔥 為 Segmentation 模型載入（不附加 Task）
        segmentation_model = load_model_manually(
            PyanNet,
            seg_dir / "config.yaml",
            seg_dir / "pytorch_model.bin",
            device,
            attach_task=True
        )
        
        # 🔥 為 Embedding 模型載入
        embedding_model = load_model_manually(
            WeSpeakerResNet34,
            emb_dir / "config.yaml",
            emb_dir / "pytorch_model.bin",
            device
        )
        
    except Exception as e:
        print(f"   ❌ 載入子模型失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 將預載入的模型注入 Pipeline
    print("\n🔄 3. 將子模型注入 SpeakerDiarization Pipeline...")
    try:
        # 核心步驟：將模型物件直接傳入建構子
        pipeline = SpeakerDiarization(
            segmentation=segmentation_model,
            embedding=embedding_model,
        )
        pipeline.to(device)
        print("   ✅ Pipeline 實例化並注入模型成功！")

    except Exception as e:
        print(f"   ❌ Pipeline 實例化失敗: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉🎉🎉 恭喜！所有測試通過，Pyannote Pipeline 已在完全離線模式下成功建立！ 🎉🎉🎉")
    print("="*50)


if __name__ == "__main__":
    test_offline_diarization_fully_manual()
