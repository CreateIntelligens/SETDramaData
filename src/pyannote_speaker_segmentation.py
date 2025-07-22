#!/usr/bin/env python3
"""
Pyannote speaker diarization and audio segmentation script
- Performs speaker diarization using pyannote
- Implements global speaker identification across multiple files using speaker embeddings.
- Segments audio based on speakers and timing
- Merges consecutive same-speaker segments
- Filters segments by duration (2-15 seconds)
- Saves segmented audio files
- Tracks processing state to avoid duplicate work
"""

import os
import sys
import re
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import librosa
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Annotation, Segment
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from speaker_database import SpeakerDatabase, migrate_from_json
from streaming_segmentation import segment_by_streaming_decision
from subtitle_driven_segmentation import segment_by_subtitle_driven

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- Database-based State Management ---

def init_speaker_database(db_path: str, legacy_json_path: str = None) -> SpeakerDatabase:
    """Initialize speaker database, migrating from JSON if needed"""
    db = SpeakerDatabase(db_path)
    
    # Check if we need to migrate from old JSON format
    if legacy_json_path and Path(legacy_json_path).exists():
        print(f"🔄 Migrating from legacy JSON format: {legacy_json_path}")
        migrate_from_json(legacy_json_path, db_path)
        print("✅ Migration completed!")
    
    return db


# --- Subtitle Parsing (unchanged) ---
def parse_subtitle_file(subtitle_path: str) -> List[Tuple[float, str]]:
    """Parse subtitle file to extract timing and text information."""
    subtitles = []
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 12:
                try:
                    if ':' in line[:12]:
                        parts = line.split(' ', 1)
                        if len(parts) >= 1:
                            timecode = parts[0]
                            text = parts[1] if len(parts) > 1 else ""
                            time_parts = timecode.split(':')
                            if len(time_parts) == 4:
                                hours, minutes, seconds, subseconds = [int(p) for p in time_parts]
                                total_seconds = hours * 3600 + minutes * 60 + seconds + subseconds / 60.0
                                if text:
                                    subtitles.append((total_seconds, text))
                except (ValueError, IndexError):
                    continue
    return subtitles


# --- Core Diarization and Embedding Functions ---

def perform_speaker_diarization(audio_path: str, pipeline: Pipeline, device: torch.device) -> Annotation:
    """Perform speaker diarization using a pre-loaded pyannote pipeline."""
    try:
        duration = librosa.get_duration(path=audio_path)
        print(f"   Analyzing audio ({duration:.1f}s) for speaker diarization...")
        
        with tqdm(total=100, desc="   Diarizing", unit="%", ncols=80) as pbar:
            diarization = None
            exception = None
            
            def diarize():
                nonlocal diarization, exception
                try:
                    diarization = pipeline(audio_path)
                except Exception as e:
                    exception = e
            
            import threading, time
            thread = threading.Thread(target=diarize)
            thread.start()
            
            progress = 0
            while thread.is_alive():
                progress = min(progress + 1, 95)
                pbar.update(1)
                time.sleep(max(0.1, duration * 0.001))
            
            thread.join()
            if exception: raise exception
            pbar.update(100 - pbar.n)
        
        return diarization
    except Exception as e:
        print(f"Error during diarization: {e}", file=sys.stderr)
        sys.exit(1)

# <<< NEW: Function to compute average embedding for a speaker
def compute_average_embedding(
    audio_path: str,
    segments: List[Segment],
    embedding_model: Inference,
    device: torch.device
) -> np.ndarray:
    """Computes the average embedding for a speaker given their speech segments."""
    embeddings = []
    
    for segment in segments:
        # Create segment dict in the format expected by pyannote
        segment_dict = {
            "audio": audio_path,
            "start": segment.start,
            "end": segment.end
        }
        
        # Extract embedding using the correct format
        try:
            with torch.no_grad():
                segment_embedding = embedding_model(segment_dict)
            
            # Convert to numpy if it's a tensor, otherwise use as-is
            if hasattr(segment_embedding, 'cpu'):
                embeddings.append(segment_embedding.cpu().numpy())
            else:
                embeddings.append(segment_embedding)
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not extract embedding for segment {segment.start:.1f}-{segment.end:.1f}s: {e}")
            continue
    
    # Return the mean of all segment embeddings
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        print(f"   ⚠️ Warning: No valid embeddings extracted")
        return None

# <<< MODIFIED: Database-based speaker identification
def find_or_register_speaker_db(
    embedding: np.ndarray,
    db: SpeakerDatabase,
    episode_num: int,
    local_label: str,
    segment_count: int,
    similarity_threshold: float = 0.85
) -> int:
    """
    Compares a speaker's embedding to the database.
    Returns the matched speaker ID or registers a new speaker.
    """
    if embedding is None:
        print(f"   ⚠️ Warning: No valid embedding for {local_label}")
        return db.add_speaker(np.zeros(512), episode_num, local_label, segment_count)
    
    # Try to find a similar speaker
    speaker_id, similarity = db.find_similar_speaker(embedding, similarity_threshold)
    
    if speaker_id is not None:
        # Update existing speaker with new episode appearance
        db.update_speaker_episode(speaker_id, episode_num, local_label, segment_count)
        return speaker_id
    else:
        # Register new speaker
        speaker_id = db.add_speaker(embedding, episode_num, local_label, segment_count)
        return speaker_id


# --- Legacy Segment Processing Functions (DEPRECATED - use subtitle_driven_segmentation.py instead) ---
# These functions are kept for backward compatibility with Traditional and Streaming modes.
# For new implementations, use the subtitle-driven segmentation approach.

def verify_speaker_similarity(
    audio_path: str,
    seg1: Tuple[float, float, str],
    seg2: Tuple[float, float, str],
    embedding_model,
    device: torch.device,
    threshold: float = 0.75
) -> bool:
    """用embedding驗證兩個segment是否為同一speaker"""
    try:
        embeddings = []
        
        # 提取兩個segment的embedding
        for start, end, _ in [seg1, seg2]:
            if end - start < 0.5:  # 至少0.5秒
                return False
            
            # Create segment dict in the format expected by pyannote
            segment_dict = {
                "audio": audio_path,
                "start": start,
                "end": end
            }
            
            try:
                with torch.no_grad():
                    embedding = embedding_model(segment_dict)
                
                # Convert to numpy if it's a tensor, otherwise use as-is
                if hasattr(embedding, 'cpu'):
                    embeddings.append(embedding.cpu().numpy())
                else:
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"   ⚠️ Warning: Could not extract embedding for verification: {e}")
                return True  # 出錯時保持原來的合併邏輯
        
        # 計算相似度
        if len(embeddings) == 2:
            similarity_tensor = F.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            )
            similarity = similarity_tensor.item() if similarity_tensor.numel() == 1 else similarity_tensor[0].item()
            return similarity > threshold
            
        return False
        
    except Exception as e:
        print(f"   ⚠️ Embedding verification failed: {e}")
        return True  # 出錯時保持原來的合併邏輯

def merge_consecutive_segments_with_verification(
    segments: List[Tuple[float, float, str]], 
    audio_path: str,
    embedding_model,
    device: torch.device,
    max_duration: float = 15.0,
    similarity_threshold: float = 0.75
) -> List[Tuple[float, float, str]]:
    """用embedding驗證後才合併consecutive segments"""
    if not segments: return []
    
    merged = []
    current_start, current_end, current_speaker = segments[0]
    
    print(f"   🔍 Verifying {len(segments)} segments before merging...")
    verified_merges = 0
    rejected_merges = 0
    
    for i in range(1, len(segments)):
        start, end, speaker = segments[i]
        
        should_merge = (
            speaker == current_speaker and 
            (end - current_start) <= max_duration
        )
        
        if should_merge:
            # 用embedding驗證是否真的是同一speaker
            current_seg = (current_start, current_end, current_speaker)
            next_seg = (start, end, speaker)
            
            if verify_speaker_similarity(audio_path, current_seg, next_seg, embedding_model, device, similarity_threshold):
                current_end = end
                verified_merges += 1
            else:
                # embedding不相似，不合併
                merged.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start, end, speaker
                rejected_merges += 1
        else:
            merged.append((current_start, current_end, current_speaker))
            current_start, current_end, current_speaker = start, end, speaker
    
    merged.append((current_start, current_end, current_speaker))
    
    print(f"   ✅ Verified merges: {verified_merges}, Rejected merges: {rejected_merges}")
    return merged

def merge_consecutive_segments(segments: List[Tuple[float, float, str]], max_duration: float = 15.0) -> List[Tuple[float, float, str]]:
    """Fallback: simple merge without verification (for compatibility)"""
    if not segments: return []
    merged = []
    current_start, current_end, current_speaker = segments[0]
    for i in range(1, len(segments)):
        start, end, speaker = segments[i]
        if speaker == current_speaker and (end - current_start) <= max_duration:
            current_end = end
        else:
            merged.append((current_start, current_end, current_speaker))
            current_start, current_end, current_speaker = start, end, speaker
    merged.append((current_start, current_end, current_speaker))
    return merged

def filter_segments_by_duration(segments: List[Tuple[float, float, str]], 
                               min_duration: float = 2.0, 
                               max_duration: float = 15.0) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    """Filter segments by duration, returning valid and invalid segments."""
    valid, invalid = [], []
    for start, end, speaker in segments:
        duration = end - start
        if min_duration <= duration <= max_duration:
            valid.append((start, end, speaker))
        else:
            invalid.append((start, end, speaker))
    return valid, invalid

# <<< LEGACY: segment_audio for Traditional mode compatibility
def segment_audio(
    audio_path: str, 
    segments: List[Tuple[float, float, str]], 
    output_dir: str, 
    subtitles: List[Tuple[float, str]], 
    episode_num: int,
    local_to_global_map: Dict[str, int]
) -> None:
    """Segment audio file, save files using the provided global speaker IDs."""
    print("   Loading audio file for segmentation...")
    audio, sr = librosa.load(audio_path, sr=None)
    os.makedirs(output_dir, exist_ok=True)
    
    tsv_data_by_dir = {}
    
    print("   Processing and saving segments with global IDs...")
    saved_count = 0
    for i, (start, end, local_speaker_label) in enumerate(tqdm(segments, desc="   Segmenting", unit="seg", ncols=80)):
        start_sample, end_sample = int(start * sr), int(end * sr)
        segment_audio_data = audio[start_sample:end_sample]
        
        if len(segment_audio_data) > 0 and np.max(np.abs(segment_audio_data)) > 0.001:
            # Use the pre-computed global speaker ID
            speaker_id = local_to_global_map[local_speaker_label]
            
            # Check for subtitle text with detailed debug info
            matching_subtitles = [(sub_time, sub_text) for sub_time, sub_text in subtitles if start <= sub_time <= end]
            segment_text = " ".join([sub_text for sub_time, sub_text in matching_subtitles]).strip()
            
            # Debug: Show segment details
            print(f"   🔍 Segment #{paragraph_id} ({start:.1f}-{end:.1f}s, {end-start:.1f}s duration):")
            print(f"       Global Speaker: {speaker_id}")
            
            if matching_subtitles:
                print(f"       ✅ Found {len(matching_subtitles)} matching subtitles:")
                for sub_time, sub_text in matching_subtitles:
                    print(f"         {sub_time:.1f}s: '{sub_text}'")
                print(f"       📝 Combined text: '{segment_text}'")
            else:
                # Show nearby subtitles for debugging
                print(f"       ❌ No matching subtitles found")
                nearby_before = [(sub_time, sub_text) for sub_time, sub_text in subtitles if start - 5 <= sub_time < start][-3:]
                nearby_after = [(sub_time, sub_text) for sub_time, sub_text in subtitles if end < sub_time <= end + 5][:3]
                
                if nearby_before:
                    print(f"       🔍 Nearby subtitles before ({start:.1f}s):")
                    for sub_time, sub_text in nearby_before:
                        print(f"         {sub_time:.1f}s: '{sub_text}' (差距: {start - sub_time:.1f}s)")
                
                if nearby_after:
                    print(f"       🔍 Nearby subtitles after ({end:.1f}s):")
                    for sub_time, sub_text in nearby_after:
                        print(f"         {sub_time:.1f}s: '{sub_text}' (差距: {sub_time - end:.1f}s)")
            
            # Skip segments without subtitle text
            if not segment_text:
                print(f"       🗑️ Skipping segment - no subtitle text found")
                continue

            chapter_id = episode_num
            paragraph_id = i + 1
            sentence_id = 1
            
            utterance_id = f"{speaker_id:03d}_{chapter_id:03d}_{paragraph_id:06d}_{sentence_id:06d}"
            
            speaker_dir = os.path.join(output_dir, f"{speaker_id:03d}")
            chapter_dir = os.path.join(speaker_dir, f"{chapter_id:03d}")
            os.makedirs(chapter_dir, exist_ok=True)
            
            audio_filename = f"{utterance_id}.wav"
            audio_path_full = os.path.join(chapter_dir, audio_filename)
            sf.write(audio_path_full, segment_audio_data, sr)
            
            text_filename = f"{utterance_id}.normalized.txt"
            text_path = os.path.join(chapter_dir, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(segment_text)
            
            dir_key = f"{speaker_id:03d}/{chapter_id:03d}"
            if dir_key not in tsv_data_by_dir:
                tsv_data_by_dir[dir_key] = []
            
            tsv_data_by_dir[dir_key].append({
                'utterance_id': utterance_id,
                'original_text': segment_text,
                'normalized_text': segment_text
            })
            saved_count += 1

    print("   Saving TSV files...")
    tsv_count = 0
    for dir_key, tsv_data in tqdm(tsv_data_by_dir.items(), desc="   Saving TSV", unit="file", ncols=80):
        full_chapter_dir = os.path.join(output_dir, dir_key)
        tsv_path = os.path.join(full_chapter_dir, f"{dir_key.replace('/', '_')}.trans.tsv")
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("utterance_id\toriginal_text\tnormalized_text\n")
            for entry in tsv_data:
                f.write(f"{entry['utterance_id']}\t{entry['original_text']}\t{entry['normalized_text']}\n")
        tsv_count += 1
    
    print(f"   💾 Saved {saved_count} audio segments and {tsv_count} TSV files")


def segment_audio_with_global_ids(
    audio_path: str, 
    segments: List[Tuple[float, float, int]], 
    output_dir: str, 
    subtitles: List[Tuple[float, str]], 
    episode_num: int
) -> None:
    """Segment audio file for streaming mode where segments already contain global IDs."""
    print("   Loading audio file for segmentation...")
    audio, sr = librosa.load(audio_path, sr=None)
    os.makedirs(output_dir, exist_ok=True)
    
    tsv_data_by_dir = {}
    
    print("   Processing and saving segments with pre-assigned global IDs...")
    saved_count = 0
    for i, (start, end, global_speaker_id) in enumerate(tqdm(segments, desc="   Segmenting", unit="seg", ncols=80)):
        start_sample, end_sample = int(start * sr), int(end * sr)
        segment_audio_data = audio[start_sample:end_sample]
        
        if len(segment_audio_data) > 0 and np.max(np.abs(segment_audio_data)) > 0.001:
            speaker_id = global_speaker_id  # Already global ID
            
            # Check for subtitle text with detailed debug info
            matching_subtitles = [(sub_time, sub_text) for sub_time, sub_text in subtitles if start <= sub_time <= end]
            segment_text = " ".join([sub_text for sub_time, sub_text in matching_subtitles]).strip()
            
            # Debug: Show segment details
            print(f"   🔍 Segment #{paragraph_id} ({start:.1f}-{end:.1f}s, {end-start:.1f}s duration):")
            print(f"       Global Speaker: {speaker_id}")
            
            if matching_subtitles:
                print(f"       ✅ Found {len(matching_subtitles)} matching subtitles:")
                for sub_time, sub_text in matching_subtitles:
                    print(f"         {sub_time:.1f}s: '{sub_text}'")
                print(f"       📝 Combined text: '{segment_text}'")
            else:
                # Show nearby subtitles for debugging
                print(f"       ❌ No matching subtitles found")
                nearby_before = [(sub_time, sub_text) for sub_time, sub_text in subtitles if start - 5 <= sub_time < start][-3:]
                nearby_after = [(sub_time, sub_text) for sub_time, sub_text in subtitles if end < sub_time <= end + 5][:3]
                
                if nearby_before:
                    print(f"       🔍 Nearby subtitles before ({start:.1f}s):")
                    for sub_time, sub_text in nearby_before:
                        print(f"         {sub_time:.1f}s: '{sub_text}' (差距: {start - sub_time:.1f}s)")
                
                if nearby_after:
                    print(f"       🔍 Nearby subtitles after ({end:.1f}s):")
                    for sub_time, sub_text in nearby_after:
                        print(f"         {sub_time:.1f}s: '{sub_text}' (差距: {sub_time - end:.1f}s)")
            
            # Skip segments without subtitle text
            if not segment_text:
                print(f"       🗑️ Skipping segment - no subtitle text found")
                continue
            
            chapter_id = episode_num
            paragraph_id = i + 1
            sentence_id = 1
            
            utterance_id = f"{speaker_id:03d}_{chapter_id:03d}_{paragraph_id:06d}_{sentence_id:06d}"
            
            speaker_dir = os.path.join(output_dir, f"{speaker_id:03d}")
            chapter_dir = os.path.join(speaker_dir, f"{chapter_id:03d}")
            os.makedirs(chapter_dir, exist_ok=True)
            
            audio_filename = f"{utterance_id}.wav"
            audio_path_full = os.path.join(chapter_dir, audio_filename)
            sf.write(audio_path_full, segment_audio_data, sr)
            
            text_filename = f"{utterance_id}.normalized.txt"
            text_path = os.path.join(chapter_dir, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(segment_text)
            
            dir_key = f"{speaker_id:03d}/{chapter_id:03d}"
            if dir_key not in tsv_data_by_dir:
                tsv_data_by_dir[dir_key] = []
            
            tsv_data_by_dir[dir_key].append({
                'utterance_id': utterance_id,
                'original_text': segment_text,
                'normalized_text': segment_text
            })
            saved_count += 1

    print("   Saving TSV files...")
    tsv_count = 0
    for dir_key, tsv_data in tqdm(tsv_data_by_dir.items(), desc="   Saving TSV", unit="file", ncols=80):
        full_chapter_dir = os.path.join(output_dir, dir_key)
        tsv_path = os.path.join(full_chapter_dir, f"{dir_key.replace('/', '_')}.trans.tsv")
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("utterance_id\toriginal_text\tnormalized_text\n")
            for entry in tsv_data:
                f.write(f"{entry['utterance_id']}\t{entry['original_text']}\t{entry['normalized_text']}\n")
        tsv_count += 1
    
    print(f"   💾 Saved {saved_count} audio segments and {tsv_count} TSV files")


def main():
    parser = argparse.ArgumentParser(description="Pyannote speaker diarization with SQLite-based global speaker identification.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("subtitle_file", help="Path to the subtitle file")
    parser.add_argument("--episode_num", type=int, required=True, help="Episode number for tracking")
    parser.add_argument("--output_dir", default="output", help="Output directory for segmented files")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Minimum segment duration")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum segment duration")
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1", help="Pyannote diarization model (name or local path)")
    parser.add_argument("--embedding_model", default="pyannote/embedding", help="Pyannote embedding model (name or local path)")
    parser.add_argument("--disable_global_matching", action="store_true", help="Disable global speaker matching (fallback to episode-based IDs)")
    parser.add_argument("--disable_merge_verification", action="store_true", help="Disable embedding verification before merging segments (faster but less accurate)")
    parser.add_argument("--merge_similarity_threshold", type=float, default=0.75, help="Similarity threshold for merge verification (lower = more strict)")
    parser.add_argument("--use_streaming_segmentation", action="store_true", help="Use streaming-style segmentation (no jump merging, only continuous)")
    parser.add_argument("--use_subtitle_driven", action="store_true", help="Use subtitle-driven segmentation (recommended) - segments based on subtitle timing with embedding merging")
    parser.add_argument("--database_path", default="speakers.db", help="Path to the SQLite speaker database")
    parser.add_argument("--legacy_json", default="processing_state.json", help="Legacy JSON state file for migration")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of an episode")
    parser.add_argument("--similarity_threshold", type=float, default=0.85, help="Cosine similarity threshold for matching speakers")
    
    args = parser.parse_args()
    
    # --- Database Initialization ---
    print("🗄️ Initializing speaker database...")
    db = init_speaker_database(args.database_path, args.legacy_json)
    
    # Check if episode already processed
    processed_episodes = db.get_processed_episodes()
    if args.episode_num in processed_episodes and not args.force:
        print(f"⚠️ Episode {args.episode_num} already processed. Use --force to re-run.")
        sys.exit(0)
    elif args.force:
        print(f"🔥 Forcing reprocessing of episode {args.episode_num}.")

    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)
    
    # Show database stats
    stats = db.get_database_stats()
    print("🚀 Starting global speaker diarization and segmentation process...")
    print(f"   Database: {stats['total_speakers']} known speakers across {stats['total_episodes']} episodes")

    # --- Model Loading ---
    print("1. Loading Pyannote models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")
    
    # 檢查是否有本地模型目錄
    project_dir = Path(__file__).parent.parent
    local_models_dir = project_dir / "models"
    if local_models_dir.exists():
        print(f"   📁 Using local models from: {local_models_dir}")
        os.environ['HF_HOME'] = str(local_models_dir / "huggingface")
        os.environ['TORCH_HOME'] = str(local_models_dir / "torch")
    
    try:
        # 載入模型時優先使用本地緩存
        model_kwargs = {}
        if local_models_dir.exists():
            model_kwargs['cache_dir'] = str(local_models_dir / "huggingface")
            print(f"   🔧 Using cache_dir: {model_kwargs['cache_dir']}")
        
        diarization_pipeline = Pipeline.from_pretrained(args.model, **model_kwargs)
        diarization_pipeline.to(device)
        
        embedding_model = Model.from_pretrained(args.embedding_model, **model_kwargs)
        if embedding_model is None:
            print(f"❌ Error: Could not load embedding model: {args.embedding_model}", file=sys.stderr)
            print("   Make sure you have HuggingFace authentication set up and access to the model.", file=sys.stderr)
            sys.exit(1)
        embedding_inference = Inference(embedding_model, window="whole").to(device)
    except Exception as e:
        print(f"❌ Error loading models: {e}", file=sys.stderr)
        if not local_models_dir.exists():
            print("   Try running: python download_models_local.py", file=sys.stderr)
        print("   Make sure you have HuggingFace authentication set up.", file=sys.stderr)
        sys.exit(1)

    # --- Main Processing Steps ---
    print("2. Parsing subtitle file...")
    subtitles = parse_subtitle_file(args.subtitle_file)
    print(f"   Found {len(subtitles)} subtitle entries.")
    
    # Choose segmentation method based on arguments
    if args.use_subtitle_driven:
        print("3. Using subtitle-driven segmentation (skipping diarization)")
        diarization = None  # Not needed for subtitle-driven approach
        print("   🎯 Using SUBTITLE-DRIVEN segmentation (recommended method)")
        print("       • Segments based on subtitle timing to ensure no text loss")
        print("       • Uses embedding similarity to merge same-speaker segments")
        segments, local_to_global_map = segment_by_subtitle_driven(
            subtitles, args.audio_file, embedding_inference.model, device,
            db, args.episode_num, args.min_duration, args.max_duration, args.similarity_threshold
        )
        print(f"   ✅ Created {len(segments)} subtitle-driven segments")
        
        # Subtitle-driven approach handles everything internally
        valid_segments = segments
        invalid_segments = []
        skip_traditional_merging = True
        skip_speaker_identification = True
        
    elif args.use_streaming_segmentation:
        print("3. Performing speaker diarization for streaming segmentation...")
        diarization = perform_speaker_diarization(args.audio_file, diarization_pipeline, device)
        print("   🔄 Using STREAMING segmentation (continuous-only merging)")
        print("       ⚠️  Note: This method may miss subtitles due to diarization timing misalignment")
        segments, local_to_global_map = segment_by_streaming_decision(
            diarization, args.audio_file, embedding_inference.model, device,
            db, args.episode_num, args.min_duration, args.max_duration, args.similarity_threshold
        )
        print(f"   Created {len(segments)} segments with streaming approach")
        
        # Skip the traditional merging step for streaming
        valid_segments = segments
        invalid_segments = []
        skip_traditional_merging = True
        skip_speaker_identification = True  # 已經在streaming中完成
        
    else:
        print("3. Performing speaker diarization for traditional segmentation...")
        diarization = perform_speaker_diarization(args.audio_file, diarization_pipeline, device)
        print("   🏛️ Using TRADITIONAL segmentation (legacy method)")
        print("       ⚠️  Note: This method may miss subtitles due to diarization timing misalignment")
        segments = [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
        segments.sort(key=lambda x: x[0])
        print(f"   Found {len(segments)} initial speaker segments for {len(diarization.labels())} local speakers.")
        skip_traditional_merging = False
        skip_speaker_identification = False
    
    # <<< MODIFIED: Database-based Global Speaker Identification Logic
    if not skip_speaker_identification:
        print("4. Identifying speakers against global database...")
        local_to_global_map = {}
    
        # Group segments by local speaker label (e.g., 'SPEAKER_00')
        local_speaker_segments = {}
        for start, end, speaker_label in segments:
            if speaker_label not in local_speaker_segments:
                local_speaker_segments[speaker_label] = []
            local_speaker_segments[speaker_label].append(Segment(start, end))

        # Process each local speaker with progress bar
        local_speakers = list(local_speaker_segments.items())
        for i, (local_label, speaker_segs) in enumerate(tqdm(local_speakers, desc="   Identifying speakers", unit="speaker", ncols=80)):
            print(f"   Processing local speaker: {local_label} ({len(speaker_segs)} segments)")
            
            # Compute average embedding for this speaker in this episode
            avg_embedding = compute_average_embedding(args.audio_file, speaker_segs, embedding_inference, device)
            
            # Debug: Show embedding info
            if avg_embedding is not None:
                print(f"     📊 Embedding shape: {avg_embedding.shape}, Mean: {np.mean(avg_embedding):.4f}, Std: {np.std(avg_embedding):.4f}")
            else:
                print(f"     ⚠️ No embedding computed for {local_label}")
            
            # Find a match in the database or register as a new speaker
            global_id = find_or_register_speaker_db(
                avg_embedding, db, args.episode_num, local_label, 
                len(speaker_segs), args.similarity_threshold
            )
            
            local_to_global_map[local_label] = global_id

        print(f"   Global speaker mapping for this episode: {local_to_global_map}")
    else:
        print("4. Using speaker identification from streaming (already completed)")

    if not skip_traditional_merging:
        if args.disable_merge_verification:
            print("5. Merging segments (fast mode, no verification)...")
            merged_segments = merge_consecutive_segments(segments, args.max_duration)
        else:
            print("5. Merging segments with embedding verification...")
            merged_segments = merge_consecutive_segments_with_verification(
                segments, 
                args.audio_file,
                embedding_inference.model,  # 使用underlying model
                device,
                args.max_duration,
                args.merge_similarity_threshold
            )
        valid_segments, invalid_segments = filter_segments_by_duration(
            merged_segments, args.min_duration, args.max_duration
        )
        print(f"   Valid segments: {len(valid_segments)}, Invalid: {len(invalid_segments)}")
    else:
        print("5. Using streaming segments (already filtered)...")
        print(f"   Valid segments: {len(valid_segments)}, Invalid: {len(invalid_segments)}")
    
    if not valid_segments:
        print("❌ No valid segments found. Exiting.")
        sys.exit(0)

    print("6. Segmenting and saving audio with global IDs...")
    if args.use_subtitle_driven or args.use_streaming_segmentation:
        # Subtitle-driven and streaming modes: segments already contain global IDs
        segment_audio_with_global_ids(
            args.audio_file, valid_segments, args.output_dir, subtitles, args.episode_num
        )
    else:
        # Traditional mode: use mapping
        segment_audio(
            args.audio_file, valid_segments, args.output_dir, subtitles, args.episode_num, local_to_global_map
        )
    
    # --- Update Database State ---
    # Mark episode as processed
    db.mark_episode_processed(args.episode_num)
    
    # Show final database stats
    final_stats = db.get_database_stats()
    print(f"💾 Database updated: {final_stats['total_speakers']} speakers, {final_stats['total_episodes']} episodes processed")

    # --- Finalization ---
    if invalid_segments:
        invalid_info_path = os.path.join(args.output_dir, f"invalid_segments_{args.episode_num:03d}.txt")
        with open(invalid_info_path, 'w', encoding='utf-8') as f:
            f.write("Invalid segments (too short or too long):\n")
            for start, end, speaker in invalid_segments:
                duration = end - start
                f.write(f"Speaker {speaker} (Global ID: {local_to_global_map.get(speaker, 'N/A')}): {start:.2f}s-{end:.2f}s ({duration:.2f}s)\n")
        print(f"ℹ️ Invalid segments info saved to: {invalid_info_path}")
    
    print("✅ Process completed successfully with global speaker IDs!")


if __name__ == "__main__":
    main()