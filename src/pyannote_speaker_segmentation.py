#!/usr/bin/env python3
"""
Pyannote Speaker Segmentation with Global Speaker Database
說話人級別分段系統：
- 合併同說話人片段，提取穩定的聲紋特徵
- 提供準確的跨集說話人識別
"""

import os
import sys
import argparse

# 自動檢測系統配置並應用最佳設定
# 加載檢測模組前，先設定基本環境變數
os.environ.update({
    'MKL_SERVICE_FORCE_INTEL': '1',
    'MKL_THREADING_LAYER': 'GNU',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'MKL_DEBUG_CPU_TYPE': '5',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'KMP_WARNINGS': 'FALSE',
    'TORCH_WARN': '0',
    'PYTORCH_DISABLE_WARNINGS': '1'
})

import torch
import numpy as np
from pathlib import Path

# 設定本地模型路徑（支援離線模式）
script_dir = Path(__file__).parent.parent
models_dir = script_dir / "models"
if models_dir.exists():
    os.environ.update({
        'HF_HOME': str(models_dir / "huggingface"),
        'TORCH_HOME': str(models_dir / "torch"),
        'HF_HUB_CACHE': str(models_dir / "huggingface" / "hub"),
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1'
    })
    print(f"🔧 使用本地模型: {models_dir}")
from typing import List, Tuple, Dict
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings

# 靜音各種煩人的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NNPACK.*")
warnings.filterwarnings("ignore", message=".*Could not initialize NNPACK.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

# 設定 logging 等級避免 C++ 層面的警告
import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# 設定 PyTorch 線程數以避免 MKL 衝突
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 簡化配置（參考能用的專案）
# 禁用問題後端
if hasattr(torch.backends, 'mkldnn'):
    torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'mkl'):
    torch.backends.mkl.enabled = False

# 檢查 GPU 可用性
if torch.cuda.is_available():
    print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    print("⚠️  GPU 不可用，使用 CPU")

# 添加 src 目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speaker_database import SpeakerDatabase
from speaker_level_segmentation import segment_by_speaker_level_approach

# Pyannote imports
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    from pyannote.core import Annotation
    import torch.nn.functional as F
except ImportError as e:
    print(f"❌ Error importing pyannote.audio: {e}")
    print("Please install pyannote.audio: pip install pyannote.audio")
    sys.exit(1)


class EmbeddingInference:
    """Embedding model wrapper for speaker verification"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from pyannote.audio import Model
            
            # 嘗試使用本地模型路徑 - 更智能的路徑檢測
            current_dir = Path.cwd()
            
            # 檢測專案根目錄
            if current_dir.name == 'src':
                project_root = current_dir.parent
            elif (current_dir / 'src').exists():
                project_root = current_dir
            else:
                project_root = Path(__file__).parent.parent
                
            # 優先使用直接下載的模型，再試快取版本  
            local_emb_path = project_root / "models" / "direct" / "embedding"
            if not local_emb_path.exists():
                local_emb_path = project_root / "models" / "huggingface" / "models--pyannote--embedding"
            
            print(f"   📁 檢查 embedding 模型路徑: {local_emb_path}")
            print(f"   📁 路徑存在: {local_emb_path.exists()}")
            
            if local_emb_path.exists():
                # 檢查是否有配置檔案
                config_file = local_emb_path / "config.yaml"
                if config_file.exists():
                    model_path = str(config_file)
                    print(f"   🔧 使用本地 embedding 模型: {local_emb_path}")
                else:
                    model_path = "pyannote/embedding"
                    print(f"   ❌ 找不到 config.yaml，改用線上 embedding 模型")
            else:
                model_path = "pyannote/embedding"
                print(f"   🌐 使用線上 embedding 模型: {model_path}")
                
            self.model = Model.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print("   ✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading embedding model: {e}")
            raise


def load_subtitles(subtitle_file: str, fps: float = 30.0) -> List[Tuple[float, str]]:
    """Load subtitles from file, supporting both HH:MM:SS:FF and seconds format"""
    subtitles = []
    
    def parse_timecode(timecode_str: str) -> float:
        """Parse timecode in HH:MM:SS:FF format to seconds"""
        try:
            # 移除 BOM 字元
            timecode_str = timecode_str.lstrip('\ufeff')
            
            # 嘗試直接解析為浮點數（秒格式）
            try:
                return float(timecode_str)
            except ValueError:
                pass
            
            # 解析 HH:MM:SS:FF 格式
            if ':' in timecode_str:
                parts = timecode_str.split(':')
                if len(parts) == 4:  # HH:MM:SS:FF
                    hours, minutes, seconds, frames = map(int, parts)
                    total_seconds = hours * 3600 + minutes * 60 + seconds + frames / fps
                    return total_seconds
                elif len(parts) == 3:  # MM:SS:FF or HH:MM:SS
                    if int(parts[0]) > 59:  # 可能是 HH:MM:SS
                        hours, minutes, seconds = map(int, parts)
                        return hours * 3600 + minutes * 60 + seconds
                    else:  # MM:SS:FF
                        minutes, seconds, frames = map(int, parts)
                        return minutes * 60 + seconds + frames / fps
            
            raise ValueError(f"Unsupported timecode format: {timecode_str}")
            
        except Exception as e:
            raise ValueError(f"Could not parse timecode '{timecode_str}': {e}")
    
    try:
        with open(subtitle_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig 自動處理 BOM
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析格式：timecode text
                    parts = line.split(' ', 1)
                    if len(parts) >= 1:
                        timecode_str = parts[0]
                        text = parts[1] if len(parts) > 1 else ""
                        
                        # 解析時間碼
                        timestamp = parse_timecode(timecode_str)
                        subtitles.append((timestamp, text))
                    else:
                        print(f"   ⚠️ Warning: Invalid format at line {line_num}: {line}")
                except ValueError as e:
                    print(f"   ⚠️ Warning: Could not parse line {line_num}: '{line}' - parts: {parts} ({e})")
                    continue
        
        print(f"   ✅ Loaded {len(subtitles)} subtitle entries")
        if subtitles:
            print(f"   📊 Time range: {subtitles[0][0]:.2f}s - {subtitles[-1][0]:.2f}s")
        return subtitles
        
    except Exception as e:
        print(f"   ❌ Error loading subtitles: {e}")
        return []


def perform_speaker_diarization(audio_file: str, pipeline: Pipeline, device: torch.device) -> Annotation:
    """Perform speaker diarization using pyannote"""
    
    print(f"   Processing audio file: {audio_file}")
    
    try:
        # 清理GPU記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 執行 diarization
        with ProgressHook() as hook:
            diarization = pipeline(audio_file, hook=hook)
        
        # 統計結果
        speakers = set()
        total_duration = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            total_duration += turn.duration
        
        print(f"   ✅ Diarization completed: {len(speakers)} speakers, {total_duration:.1f}s total speech")
        
        return diarization
        
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        raise


def segment_audio_files(
    segments: List[Tuple[float, float, int]], 
    audio_path: str, 
    output_dir: str, 
    subtitles: List[Tuple[float, str]], 
    episode_num: int
) -> None:
    """Segment audio file and save with global speaker IDs"""
    
    print("4. Segmenting and saving audio files...")
    
    # 載入音檔
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"   ✅ Audio loaded: {len(audio)/sr:.1f}s, {sr}Hz")
    except Exception as e:
        print(f"   ❌ Error loading audio: {e}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 建立字幕查找字典
    subtitle_dict = {timestamp: text for timestamp, text in subtitles}
    
    saved_count = 0
    
    for i, (start, end, speaker_id) in enumerate(tqdm(segments, desc="   Saving segments", unit="seg", ncols=80)):
        try:
            # 提取音檔片段
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # 尋找對應的字幕
            segment_text = ""
            for timestamp, text in subtitles:
                if start <= timestamp < end:
                    segment_text += text + " "
            segment_text = segment_text.strip()
            
            if not segment_text:
                continue  # 跳過沒有字幕的片段
            
            # 建立檔案路徑
            chapter_id = episode_num
            paragraph_id = i + 1
            sentence_id = 1
            
            utterance_id = f"{speaker_id:03d}_{chapter_id:03d}_{paragraph_id:06d}_{sentence_id:06d}"
            
            speaker_dir = os.path.join(output_dir, f"{speaker_id:03d}")
            chapter_dir = os.path.join(speaker_dir, f"{chapter_id:03d}")
            os.makedirs(chapter_dir, exist_ok=True)
            
            # 儲存音檔
            audio_filename = f"{utterance_id}.wav"
            audio_filepath = os.path.join(chapter_dir, audio_filename)
            sf.write(audio_filepath, segment_audio, sr)
            
            # 儲存字幕
            text_filename = f"{utterance_id}.normalized.txt"
            text_filepath = os.path.join(chapter_dir, text_filename)
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(segment_text)
            
            saved_count += 1
            
        except Exception as e:
            print(f"   ⚠️ Error saving segment {i}: {e}")
            continue
    
    print(f"   ✅ Saved {saved_count} audio segments to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pyannote Speaker Segmentation with Global Speaker Database")
    parser.add_argument("audio_file", help="Input audio file")
    parser.add_argument("subtitle_file", help="Input subtitle file (timestamp text format)")
    parser.add_argument("--episode_num", type=int, required=True, help="Episode number")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum segment duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum segment duration (seconds)")
    parser.add_argument("--similarity_threshold", type=float, default=0.40, help="Cosine similarity threshold for matching speakers")
    parser.add_argument("--voice_activity_threshold", type=float, default=0.1, help="Voice activity threshold for hybrid segmentation (0.0-1.0)")
    parser.add_argument("--min_speaker_duration", type=float, default=5.0, help="Minimum total duration for a speaker to be considered (seconds)")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if episode already processed")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # 設定裝置
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 檢查檔案
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.subtitle_file):
        print(f"❌ Subtitle file not found: {args.subtitle_file}")
        sys.exit(1)
    
    # 初始化資料庫
    print("1. Initializing speaker database...")
    db = SpeakerDatabase()
    
    # 檢查是否已處理過
    processed_episodes = db.get_processed_episodes()
    if args.episode_num in processed_episodes and not args.force:
        print(f"   ⚠️ Episode {args.episode_num} already processed. Use --force to reprocess.")
        sys.exit(0)
    
    # 載入字幕
    print("2. Loading subtitles...")
    subtitles = load_subtitles(args.subtitle_file)
    if not subtitles:
        print("❌ No valid subtitles loaded")
        sys.exit(1)
    
    # 初始化模型
    print("3. Initializing models...")
    
    # Diarization pipeline
    try:
        # 嘗試使用本地模型路徑 - 更智能的路徑檢測
        current_dir = Path.cwd()
        
        # 檢測專案根目錄
        if current_dir.name == 'src':
            project_root = current_dir.parent
        elif (current_dir / 'src').exists():
            project_root = current_dir
        else:
            project_root = Path(__file__).parent.parent
            
        # 優先使用直接下載的模型，再試快取版本
        local_diar_path = project_root / "models" / "direct" / "speaker-diarization-3.1"
        if not local_diar_path.exists():
            local_diar_path = project_root / "models" / "huggingface" / "models--pyannote--speaker-diarization-3.1"
        
        print(f"   📁 檢查專案根目錄: {project_root}")
        print(f"   📁 檢查模型路徑: {local_diar_path}")
        print(f"   📁 路徑存在: {local_diar_path.exists()}")
        
        if local_diar_path.exists():
            print(f"   🔧 使用本地 diarization 模型: {local_diar_path}")
            
            # 🔥 完全使用本地路徑，不使用 repo ID
            config_file = local_diar_path / "config.yaml"
            if config_file.exists():
                try:
                    # 直接從本地配置檔案載入
                    diarization_pipeline = Pipeline.from_pretrained(str(config_file))
                    print("   ✅ 從本地配置檔案載入成功")
                except Exception as e:
                    print(f"   ❌ 本地配置載入失敗: {e}")
                    print("   💡 可能需要修改 config.yaml 中的模型路徑引用")
                    raise
            else:
                raise Exception(f"找不到配置檔案: {config_file}")
                
        else:
            print("   🌐 使用線上 diarization 模型")
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            
        diarization_pipeline = diarization_pipeline.to(device)
        print("   ✅ Diarization pipeline loaded")
    except Exception as e:
        print(f"   ❌ Error loading diarization pipeline: {e}")
        print(f"   📁 檢查路徑: {script_dir / 'models'}")
        sys.exit(1)
    
    # Embedding model
    try:
        embedding_inference = EmbeddingInference(device)
        print("   ✅ Embedding model loaded")
    except Exception as e:
        print(f"   ❌ Error loading embedding model: {e}")
        sys.exit(1)
    
    # 執行 diarization
    print("4. Performing speaker diarization...")
    diarization = perform_speaker_diarization(args.audio_file, diarization_pipeline, device)
    
    # 執行說話人級別分段
    print("   🎯 使用說話人級別分段模式")
    segments, local_to_global_map = segment_by_speaker_level_approach(
        diarization, subtitles, args.audio_file, embedding_inference.model, device,
        db, args.episode_num, args.min_duration, args.max_duration, args.similarity_threshold,
        args.min_speaker_duration
    )
    print(f"   ✅ 創建了 {len(segments)} 個說話人級別分段")
    
    if not segments:
        print("❌ No valid segments created")
        sys.exit(1)
    
    # 分割並儲存音檔
    segment_audio_files(segments, args.audio_file, args.output_dir, subtitles, args.episode_num)
    
    # 標記集數為已處理
    db.mark_episode_processed(args.episode_num)
    
    # 顯示統計資訊
    print("\n" + "="*60)
    print("📊 Processing Summary")
    print("="*60)
    print(f"Episode: {args.episode_num}")
    print(f"Total Segments: {len(segments)}")
    print(f"Unique Speakers: {len(set(seg[2] for seg in segments))}")
    print(f"Similarity Threshold: {args.similarity_threshold}")
    print(f"Min Speaker Duration: {args.min_speaker_duration}s")
    
    if local_to_global_map:
        print(f"\nSpeaker Mapping:")
        for local_label, global_id in sorted(local_to_global_map.items()):
            print(f"  {local_label} → Global Speaker {global_id}")
    
    # 顯示資料庫統計
    db_stats = db.get_database_stats()
    print(f"\nDatabase Stats:")
    print(f"  Total Speakers: {db_stats['total_speakers']}")
    print(f"  Total Episodes: {db_stats['total_episodes']}")
    print(f"  Total Segments: {db_stats['total_segments']}")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
