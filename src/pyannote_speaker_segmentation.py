#!/usr/bin/env python3
"""
Pyannote Speaker Segmentation with Global Speaker Database
說話人級別分段系統 - 使用官方正規離線方法
"""

import os
import sys
from pathlib import Path

# 取得專案根目錄
project_root = Path(__file__).parent.parent

# 載入 .env 檔案
def load_env_file():
    """載入 .env 檔案中的環境變數"""
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"📁 載入環境變數: {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
                    if key in ['SIMILARITY_THRESHOLD', 'MIN_SPEAKER_DURATION', 'VOICE_ACTIVITY_THRESHOLD']:
                        print(f"  ✅ {key}={value}")
    else:
        print(f"⚠️ 找不到 .env 檔案: {env_file}")

# 載入環境變數
load_env_file()

# 匯入正規離線載入模組
try:
    from offline_pipeline import load_offline_pipeline
    OFFICIAL_OFFLINE_AVAILABLE = True
    print("🎯 使用官方正規離線方法")
except ImportError:
    OFFICIAL_OFFLINE_AVAILABLE = False
    print("⚠️ 正規離線模組不可用，使用備用方法")
    
    # 備用：舊的離線設定
    models_dir = project_root / "models" / "huggingface"
    
    # 強制離線模式環境變數
    os.environ.update({
        'TRANSFORMERS_OFFLINE': '1',
        'HUGGINGFACE_HUB_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1'
    })
    
    print(f"🔧 模型目錄: {models_dir}")
    print(f"🔧 完全離線模式: 已啟用")

# 記憶體優化設定
os.environ.update({
    'MKL_SERVICE_FORCE_INTEL': '1',
    'MKL_THREADING_LAYER': 'GNU',
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'TORCH_WARN': '0'
})

import torch
import numpy as np
import argparse
from typing import List, Tuple, Dict
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
import logging
import gc

# 靜音警告
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# PyTorch 記憶體優化
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# GPU 記憶體設定
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"📊 GPU 記憶體: {gpu_memory:.1f}GB (自動管理)")
else:
    print("⚠️ 使用 CPU")

# 添加 src 目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加專案根目錄到路徑（用於導入 fix_symlinks）
sys.path.append(str(project_root))

from speaker_database import SpeakerDatabase
from speaker_level_segmentation import segment_by_speaker_level_approach

# Pyannote imports
try:
    from pyannote.audio import Pipeline, Model
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    from pyannote.core import Annotation
    print("✅ Pyannote 模組載入成功")
except ImportError as e:
    print(f"❌ Pyannote 載入失敗: {e}")
    sys.exit(1)


class EmbeddingInference:
    """Embedding model wrapper - 使用正規離線方法"""

    def __init__(self, device: torch.device, pipeline=None):
        self.device = device
        self.model = None
        self.pipeline = pipeline
        self._load_model()

    def _load_model(self):
        """載入 embedding 模型"""
        try:
            if OFFICIAL_OFFLINE_AVAILABLE and self.pipeline:
                # 🎯 使用正規離線方法：從 pipeline 中取得 embedding 模型
                print("📁 使用正規離線方法載入 embedding 模型...")
                print(f"🔍 Pipeline 屬性: {[attr for attr in dir(self.pipeline) if 'embed' in attr.lower()]}")
                
                # 嘗試不同的屬性名稱
                for attr_name in ['_embedding', 'embedding', '_embedding_model', 'embedding_model']:
                    if hasattr(self.pipeline, attr_name):
                        embedding_model = getattr(self.pipeline, attr_name)
                        if embedding_model is not None:
                            self.model = embedding_model
                            print(f"✅ 從 Pipeline.{attr_name} 取得 Embedding 模型成功")
                            return
                
                print("⚠️ Pipeline 中沒有找到 embedding 模型，使用備用方法...")
            
            # 備用方法：傳統載入
            print("📁 使用備用方法載入 embedding 模型...")
            from pyannote.audio import Model
            
            if OFFICIAL_OFFLINE_AVAILABLE:
                # 嘗試從正規離線模型檔案載入
                project_root = Path(__file__).parent.parent
                model_path = project_root / "models" / "pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
                
                if model_path.exists():
                    print(f"📁 載入正規離線模型: {model_path}")
                    try:
                        from pyannote.audio import Model
                        # 嘗試直接載入 .bin 檔案
                        self.model = Model.from_pretrained(str(model_path)).to(self.device)
                        self.model.eval()
                        print("✅ 正規離線 Embedding 模型載入成功")
                        return
                    except Exception as e:
                        print(f"⚠️ 直接載入 .bin 檔案失敗: {e}")
                        # 繼續使用備用方法
            
            # 最後的備用方法
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models" / "huggingface"
            
            print(f"📁 快取目錄: {models_dir}")
            self.model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                cache_dir=str(models_dir),
                local_files_only=True,
                use_auth_token=None
            ).to(self.device)
            self.model.eval()
            print("✅ Embedding 模型載入成功")

        except Exception as e:
            print(f"❌ Embedding 模型載入失敗: {e}")
            # 如果是正規離線環境，這可能不是致命錯誤
            if OFFICIAL_OFFLINE_AVAILABLE and self.pipeline:
                print("💡 正規離線環境中，embedding 功能已整合在 Pipeline 中")
                self.model = None  # 設為 None，後續會檢查
            else:
                raise


def load_subtitles(subtitle_file: str, fps: float = 30.0) -> List[Tuple[float, str]]:
    """載入字幕檔案"""
    subtitles = []

    def parse_timecode(timecode_str: str) -> float:
        """解析時間碼"""
        try:
            timecode_str = timecode_str.lstrip('\ufeff')

            try:
                return float(timecode_str)
            except ValueError:
                pass

            if ':' in timecode_str:
                parts = timecode_str.split(':')
                if len(parts) == 4:  # HH:MM:SS:FF
                    hours, minutes, seconds, frames = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds + frames / fps
                elif len(parts) == 3:
                    if int(parts[0]) > 59:  # HH:MM:SS
                        hours, minutes, seconds = map(int, parts)
                        return hours * 3600 + minutes * 60 + seconds
                    else:  # MM:SS:FF
                        minutes, seconds, frames = map(int, parts)
                        return minutes * 60 + seconds + frames / fps

            raise ValueError(f"不支援的時間碼格式: {timecode_str}")

        except Exception as e:
            raise ValueError(f"無法解析時間碼 '{timecode_str}': {e}")

    try:
        with open(subtitle_file, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split(' ', 1)
                    if len(parts) >= 1:
                        timecode_str = parts[0]
                        text = parts[1] if len(parts) > 1 else ""
                        timestamp = parse_timecode(timecode_str)
                        subtitles.append((timestamp, text))
                except ValueError as e:
                    print(f"⚠️ 第 {line_num} 行解析失敗: {e}")
                    continue

        print(f"✅ 載入 {len(subtitles)} 個字幕")
        return subtitles

    except Exception as e:
        print(f"❌ 字幕載入失敗: {e}")
        return []


def perform_speaker_diarization(audio_file: str, pipeline, device: torch.device) -> Annotation:
    """執行說話人分離"""
    print(f"🎵 處理音檔: {audio_file}")

    try:
        # 簡單記憶體清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 顯示記憶體狀態
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU 記憶體: 已分配 {allocated:.2f}GB, 快取 {cached:.2f}GB")

        print("🚀 執行 diarization...")
        with ProgressHook() as hook:
            diarization = pipeline(audio_file, hook=hook)

        # 統計結果
        speakers = set()
        total_duration = 0.0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            total_duration += turn.duration

        print(f"✅ 完成: {len(speakers)} 個說話人, {total_duration:.1f}s")
        
        # 顯示最終記憶體狀態
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 完成後 GPU 記憶體: 已分配 {allocated:.2f}GB, 快取 {cached:.2f}GB")
        
        return diarization

    except Exception as e:
        print(f"❌ Diarization 失敗: {e}")
        raise




def segment_audio_files(segments, audio_path, output_dir, subtitles, episode_num):
    """分割並儲存音檔"""
    print("📁 分割音檔...")

    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"✅ 音檔載入: {len(audio)/sr:.1f}s, {sr}Hz")
    except Exception as e:
        print(f"❌ 音檔載入失敗: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    for i, (start, end, speaker_id) in enumerate(tqdm(segments, desc="儲存片段")):
        try:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            # 尋找字幕
            segment_text = ""
            for timestamp, text in subtitles:
                if start <= timestamp < end:
                    segment_text += text + " "
            segment_text = segment_text.strip()

            if not segment_text:
                continue

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
            print(f"⚠️ 片段 {i} 儲存失敗: {e}")
            continue

    print(f"✅ 儲存 {saved_count} 個片段到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pyannote Speaker Segmentation")
    parser.add_argument("audio_file", help="音檔路徑")
    parser.add_argument("subtitle_file", help="字幕檔路徑")
    parser.add_argument("--episode_num", type=int, required=True, help="集數")
    parser.add_argument("--output_dir", default="output", help="輸出目錄")
    parser.add_argument("--min_duration", type=float, default=1.0, help="最小片段長度")
    parser.add_argument("--max_duration", type=float, default=15.0, help="最大片段長度")
    parser.add_argument("--similarity_threshold", type=float, 
                        default=float(os.environ.get('SIMILARITY_THRESHOLD', '0.40')), 
                        help="相似度閾值")
    parser.add_argument("--voice_activity_threshold", type=float, 
                        default=float(os.environ.get('VOICE_ACTIVITY_THRESHOLD', '0.1')), 
                        help="語音活動閾值")
    parser.add_argument("--min_speaker_duration", type=float, 
                        default=float(os.environ.get('MIN_SPEAKER_DURATION', '5.0')), 
                        help="最小說話人時長")
    parser.add_argument("--force", action="store_true", help="強制重新處理")
    parser.add_argument("--device", choices=["cpu", "cuda"], 
                       default="cuda" if torch.cuda.is_available() else "cpu", help="裝置")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🔧 使用裝置: {device}")

    # 檢查檔案
    if not os.path.exists(args.audio_file):
        print(f"❌ 音檔不存在: {args.audio_file}")
        sys.exit(1)

    if not os.path.exists(args.subtitle_file):
        print(f"❌ 字幕檔不存在: {args.subtitle_file}")
        sys.exit(1)

    # 初始化資料庫
    print("1. 初始化資料庫...")
    db = SpeakerDatabase()

    # 檢查是否已處理
    processed_episodes = db.get_processed_episodes()
    if args.episode_num in processed_episodes and not args.force:
        print(f"⚠️ 集數 {args.episode_num} 已處理過，使用 --force 重新處理")
        sys.exit(0)

    # 載入字幕
    print("2. 載入字幕...")
    subtitles = load_subtitles(args.subtitle_file)
    if not subtitles:
        print("❌ 字幕載入失敗")
        sys.exit(1)

    # 跳過符號連結修復（使用正規離線方法不需要）
    print("3. 正規離線方法不需要符號連結修復，跳過此步驟...")

    # 載入 diarization pipeline
    print("4. 載入 diarization pipeline...")
    try:
        if OFFICIAL_OFFLINE_AVAILABLE:
            # 🎯 使用官方正規離線方法
            print("使用官方正規離線載入方法...")
            diarization_pipeline, device_type = load_offline_pipeline()
            print(f"✅ 正規離線 Pipeline 載入成功 ({device_type.upper()})")
            
            # 如果需要指定不同設備
            if str(device) != device_type:
                diarization_pipeline.to(device)
                print(f"📱 Pipeline 已移至 {device}")
                
        else:
            # 🔄 備用方法：使用快取目錄
            print(f"📁 使用備用方法，快取目錄: {models_dir}")
            from pyannote.audio import Pipeline
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                cache_dir=str(models_dir),
                use_auth_token=None
            ).to(device)
            print("✅ 備用方法 Pipeline 載入成功")

        # 記憶體清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Pipeline 載入失敗: {e}")
        if OFFICIAL_OFFLINE_AVAILABLE:
            print("💡 正規方法載入失敗，請檢查 models/config.yaml 和模型檔案")
        else:
            print(f"💡 快取目錄: {models_dir}")
        sys.exit(1)

    # 執行 diarization
    print("5. 執行 speaker diarization...")
    diarization = perform_speaker_diarization(args.audio_file, diarization_pipeline, device)

    # 載入 embedding model (在釋放 pipeline 前)
    print("6. 載入 embedding 模型...")
    
    try:
        # 🎯 在所有模式下都載入 embedding 模型，因為後續處理需要用到
        embedding_inference = EmbeddingInference(device, pipeline=diarization_pipeline if OFFICIAL_OFFLINE_AVAILABLE else None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Embedding 模型載入失敗: {e}")
        sys.exit(1)

    # 釋放 pipeline 記憶體
    print("🧹 釋放 pipeline 記憶體...")
    del diarization_pipeline
    for _ in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 執行說話人級別分段
    print("7. 執行說話人級別分段...")
    embedding_model = embedding_inference.model if embedding_inference else None
    segments, local_to_global_map = segment_by_speaker_level_approach(
        diarization, subtitles, args.audio_file, embedding_model, device,
        db, args.episode_num, args.min_duration, args.max_duration,
        args.similarity_threshold, args.min_speaker_duration
    )
    print(f"✅ 建立 {len(segments)} 個分段")

    if not segments:
        print("❌ 沒有有效分段")
        sys.exit(1)

    # 分割音檔
    segment_audio_files(segments, args.audio_file, args.output_dir, subtitles, args.episode_num)

    # 標記為已處理
    db.mark_episode_processed(args.episode_num)

    # 顯示統計
    print("\n" + "="*60)
    print("📊 處理摘要")
    print("="*60)
    print(f"集數: {args.episode_num}")
    print(f"總分段: {len(segments)}")
    print(f"說話人數: {len(set(seg[2] for seg in segments))}")

    if local_to_global_map:
        print(f"\n說話人對應:")
        for local_label, global_id in sorted(local_to_global_map.items()):
            print(f"  {local_label} → 全域說話人 {global_id}")

    db_stats = db.get_database_stats()
    print(f"\n資料庫統計:")
    print(f"  總說話人: {db_stats['total_speakers']}")
    print(f"  總集數: {db_stats['total_episodes']}")
    print(f"  總分段: {db_stats['total_segments']}")

    print(f"\n✅ 處理完成！")
    print(f"輸出目錄: {args.output_dir}")


if __name__ == "__main__":
    main()
