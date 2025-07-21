#!/usr/bin/env python3
"""
Pyannote speaker diarization and audio segmentation script
- Performs speaker diarization using pyannote
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
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import argparse
from tqdm import tqdm
import torch

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- State Management Functions ---

def load_state(state_path: str) -> Dict:
    """Loads the processing state from a JSON file."""
    if os.path.exists(state_path):
        with open(state_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "last_used_speaker_id": -1, 
        "processed_episodes": [],
        "episode_speaker_ranges": {}  # episode_num -> {"start": id, "end": id, "mapping": {...}}
    }

def save_state(state_path: str, state: Dict):
    """Saves the processing state to a JSON file."""
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    print(f"💾 State saved to {state_path}")


def parse_subtitle_file(subtitle_path: str) -> List[Tuple[float, str]]:
    """Parse subtitle file to extract timing and text information."""
    subtitles = []
    
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 12:  # Minimum length for timecode
                try:
                    # Check if line starts with timecode pattern HH:MM:SS:FF
                    if ':' in line[:12]:
                        # Split by spaces, first part should be timecode
                        parts = line.split(' ', 1)
                        if len(parts) >= 1:
                            timecode = parts[0]
                            text = parts[1] if len(parts) > 1 else ""
                            
                            # Parse timecode HH:MM:SS:FF (assuming 25fps)
                            time_parts = timecode.split(':')
                            if len(time_parts) == 4:
                                hours = int(time_parts[0])
                                minutes = int(time_parts[1])
                                seconds = int(time_parts[2])
                                frames = int(time_parts[3])
                                
                                # Convert to seconds (assuming 25fps)
                                total_seconds = hours * 3600 + minutes * 60 + seconds + frames / 25.0
                                if text:  # Only add if there's text
                                    subtitles.append((total_seconds, text))
                except (ValueError, IndexError):
                    continue
    
    return subtitles


def perform_speaker_diarization(audio_path: str, model_name: str = "pyannote/speaker-diarization-3.1") -> Annotation:
    """Perform speaker diarization using pyannote."""
    try:
        # Load the pipeline
        print("   Loading pyannote model...")
        pipeline = Pipeline.from_pretrained(model_name)
        
        # Check for GPU availability and use it if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            pipeline = pipeline.to(device)
            print(f"   🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("   💻 Using CPU (consider using GPU for faster processing)")
        
        # Get audio duration for progress tracking
        import librosa
        duration = librosa.get_duration(path=audio_path)
        
        # Perform diarization with progress
        print(f"   Analyzing audio ({duration:.1f}s) for speaker diarization...")
        
        # Create a progress bar that updates based on estimated time
        with tqdm(total=100, desc="   Diarizing", unit="%", ncols=70) as pbar:
            # Start diarization in a separate thread for progress updates
            import threading
            import time
            
            diarization = None
            exception = None
            
            def diarize():
                nonlocal diarization, exception
                try:
                    diarization = pipeline(audio_path)
                except Exception as e:
                    exception = e
            
            # Start diarization thread
            thread = threading.Thread(target=diarize)
            thread.start()
            
            # Update progress bar while diarization is running
            progress = 0
            while thread.is_alive():
                progress = min(progress + 1, 95)  # Don't go to 100% until done
                pbar.update(1)
                time.sleep(max(0.1, duration * 0.001))  # Adaptive sleep based on audio length
            
            # Wait for completion
            thread.join()
            
            if exception:
                raise exception
            
            # Complete progress bar
            pbar.update(100 - progress)
        
        return diarization
    except Exception as e:
        print(f"Error during diarization: {e}")
        print("Make sure you have the required pyannote models installed and authentication set up.")
        sys.exit(1)


def merge_consecutive_segments(segments: List[Tuple[float, float, str]], max_duration: float = 15.0) -> List[Tuple[float, float, str]]:
    """Merge consecutive segments from the same speaker if total duration <= max_duration."""
    if not segments:
        return []
    
    merged = []
    current_start, current_end, current_speaker = segments[0]
    
    for i in range(1, len(segments)):
        start, end, speaker = segments[i]
        
        # Check if same speaker and total duration would be within limit
        if (speaker == current_speaker and 
            (end - current_start) <= max_duration):
            # Merge by extending the current segment
            current_end = end
        else:
            # Save current segment and start new one
            merged.append((current_start, current_end, current_speaker))
            current_start, current_end, current_speaker = start, end, speaker
    
    # Add the last segment
    merged.append((current_start, current_end, current_speaker))
    
    return merged


def filter_segments_by_duration(segments: List[Tuple[float, float, str]], 
                               min_duration: float = 2.0, 
                               max_duration: float = 15.0) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    """Filter segments by duration, returning valid and invalid segments."""
    valid_segments = []
    invalid_segments = []
    
    for start, end, speaker in segments:
        duration = end - start
        if min_duration <= duration <= max_duration:
            valid_segments.append((start, end, speaker))
        else:
            invalid_segments.append((start, end, speaker))
    
    return valid_segments, invalid_segments


def segment_audio(audio_path: str, segments: List[Tuple[float, float, str]], output_dir: str, 
                 subtitles: List[Tuple[float, str]], episode_num: int, state: Dict) -> Tuple[Dict, Dict, int]:
    """Segment audio file, save files, and return new state information."""
    print("   Loading audio file...")
    with tqdm(desc="   Loading audio", unit="MB", ncols=70) as pbar:
        audio, sr = librosa.load(audio_path, sr=None)
        pbar.update(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tsv_data_by_dir = {}
    speaker_mapping = {}
    
    # Get the next available speaker ID from the state
    base_speaker_id = state.get("last_used_speaker_id", -1) + 1
    new_max_speaker_id = base_speaker_id - 1

    print("   Processing and saving segments...")
    saved_count = 0
    for i, (start, end, speaker) in enumerate(tqdm(segments, desc="   Segmenting", unit="seg", ncols=70)):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio_data = audio[start_sample:end_sample]
        
        if len(segment_audio_data) > 0 and np.max(np.abs(segment_audio_data)) > 0.001:
            if speaker not in speaker_mapping:
                # Assign a new, unique speaker ID
                speaker_mapping[speaker] = base_speaker_id + len(speaker_mapping)
            
            speaker_id = speaker_mapping[speaker]
            new_max_speaker_id = max(new_max_speaker_id, speaker_id)

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
            
            segment_text = ""
            if subtitles:
                for sub_time, sub_text in subtitles:
                    if start <= sub_time <= end:
                        segment_text += sub_text + " "
            segment_text = segment_text.strip()
            
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
            
            # Only print every 50th segment to avoid spam
            if saved_count % 50 == 0:
                print(f"   Saved {saved_count} segments...")
    
    print("   Saving TSV files...")
    tsv_count = 0
    for dir_key, tsv_data in tqdm(tsv_data_by_dir.items(), desc="   Saving TSV", unit="file", ncols=70):
        full_chapter_dir = os.path.join(output_dir, dir_key)
        tsv_path = os.path.join(full_chapter_dir, f"{dir_key.replace('/', '_')}.trans.tsv")
        
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("utterance_id\toriginal_text\tnormalized_text\n")
            for entry in tsv_data:
                f.write(f"{entry['utterance_id']}\t{entry['original_text']}\t{entry['normalized_text']}\n")
        
        tsv_count += 1
    
    print(f"   💾 Saved {saved_count} audio segments and {tsv_count} TSV files")
    
    print(f"📋 Speaker mapping for this run: {speaker_mapping}")
    
    # Show speaker ID range for this episode
    if speaker_mapping:
        speaker_ids = list(speaker_mapping.values())
        min_id = min(speaker_ids)
        max_id = max(speaker_ids)
        print(f"🔢 Episode speaker ID range: {min_id} - {max_id} ({len(speaker_ids)} speakers)")
    
    return tsv_data_by_dir, speaker_mapping, new_max_speaker_id


def main():
    parser = argparse.ArgumentParser(description="Pyannote speaker diarization and audio segmentation with state tracking.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("subtitle_file", help="Path to the subtitle file")
    parser.add_argument("--episode_num", type=int, required=True, help="Episode number for tracking and chapter ID")
    parser.add_argument("--output_dir", default="segmented_audio", help="Output directory for segmented files")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Minimum segment duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum segment duration in seconds")
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1", help="Pyannote model to use")
    parser.add_argument("--state_file", default="processing_state.json", help="Path to the processing state file")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of an already processed episode")
    
    args = parser.parse_args()
    
    # --- State and File Checks ---
    state = load_state(args.state_file)
    
    if args.episode_num in state["processed_episodes"] and not args.force:
        print(f"⚠️ Episode {args.episode_num} has already been processed. Use --force to re-run.")
        print(f"Processed episodes: {state['processed_episodes']}")
        sys.exit(0)
    elif args.episode_num in state["processed_episodes"] and args.force:
        print(f"🔥 Forcing reprocessing of episode {args.episode_num}.")

    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.subtitle_file):
        print(f"❌ Error: Subtitle file not found: {args.subtitle_file}")
        sys.exit(1)
    
    print("🚀 Starting speaker diarization and segmentation process...")
    print(f"Current state: Last used speaker ID: {state['last_used_speaker_id']}, Processed episodes: {len(state['processed_episodes'])}")

    # --- Main Processing Steps ---
    print("1. Parsing subtitle file...")
    subtitles = parse_subtitle_file(args.subtitle_file)
    print(f"   Found {len(subtitles)} subtitle entries")
    
    print("2. Performing speaker diarization...")
    diarization = perform_speaker_diarization(args.audio_file, args.model)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    print(f"   Found {len(segments)} initial speaker segments")
    
    segments.sort(key=lambda x: x[0])
    
    print("3. Merging consecutive same-speaker segments...")
    merged_segments = merge_consecutive_segments(segments, args.max_duration)
    print(f"   After merging: {len(merged_segments)} segments")
    
    print("4. Filtering segments by duration...")
    valid_segments, invalid_segments = filter_segments_by_duration(
        merged_segments, args.min_duration, args.max_duration
    )
    print(f"   Valid segments: {len(valid_segments)}")
    print(f"   Invalid segments (too short/long): {len(invalid_segments)}")
    
    if not valid_segments:
        print("❌ No valid segments found after filtering. Exiting.")
        sys.exit(0)

    print("5. Segmenting and saving audio files...")
    _, speaker_mapping, new_max_speaker_id = segment_audio(
        args.audio_file, valid_segments, args.output_dir, subtitles, args.episode_num, state
    )
    
    # --- Update and Save State ---
    episode_start_id = state["last_used_speaker_id"] + 1
    state["last_used_speaker_id"] = new_max_speaker_id
    if args.episode_num not in state["processed_episodes"]:
        state["processed_episodes"].append(args.episode_num)
        state["processed_episodes"].sort()
    
    # Record speaker ID range for this episode
    if "episode_speaker_ranges" not in state:
        state["episode_speaker_ranges"] = {}
    
    state["episode_speaker_ranges"][str(args.episode_num)] = {
        "start": episode_start_id,
        "end": new_max_speaker_id,
        "mapping": speaker_mapping
    }

    save_state(args.state_file, state)

    # --- Finalization ---
    if invalid_segments:
        invalid_info_path = os.path.join(args.output_dir, "invalid_segments.txt")
        with open(invalid_info_path, 'w', encoding='utf-8') as f:
            f.write("Invalid segments (too short or too long):\n")
            for start, end, speaker in invalid_segments:
                duration = end - start
                f.write(f"Speaker {speaker}: {start:.2f}s-{end:.2f}s ({duration:.2f}s)\n")
        print(f"ℹ️ Invalid segments info saved to: {invalid_info_path}")
    
    print("✅ Process completed successfully!")


if __name__ == "__main__":
    main()
