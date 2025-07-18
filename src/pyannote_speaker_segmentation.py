#!/usr/bin/env python3
"""
Pyannote speaker diarization and audio segmentation script
- Performs speaker diarization using pyannote
- Segments audio based on speakers and timing
- Merges consecutive same-speaker segments
- Filters segments by duration (2-15 seconds)
- Saves segmented audio files
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import librosa
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import argparse


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
        import torch
        
        # Load the pipeline
        pipeline = Pipeline.from_pretrained(model_name)
        
        # Check for GPU availability and use it if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            pipeline = pipeline.to(device)
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU (consider using GPU for faster processing)")
        
        # Perform diarization
        diarization = pipeline(audio_path)
        
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
                 subtitles: List[Tuple[float, str]] = None, episode_num: int = None):
    """Segment audio file based on speaker segments and save individual files with LibriTTS format."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare TSV data for each speaker/chapter combination
    tsv_data_by_dir = {}
    
    # Calculate starting speaker ID based on episode number
    # Assuming max 10 speakers per episode for now
    speaker_mapping = {}
    speakers_per_episode = 10
    base_speaker_id = (episode_num - 1) * speakers_per_episode if episode_num else 0
    
    for i, (start, end, speaker) in enumerate(segments):
        # Convert time to samples
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Extract segment
        segment_audio = audio[start_sample:end_sample]
        
        # Check if segment is not empty/silent
        if len(segment_audio) > 0 and np.max(np.abs(segment_audio)) > 0.001:
            # Map speaker to numeric ID
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = base_speaker_id + len(speaker_mapping)
            
            speaker_id = speaker_mapping[speaker]
            
            # Generate LibriTTS format filename
            # Format: {speaker_id}_{chapter_id}_{paragraph_id}_{sentence_id}
            chapter_id = episode_num if episode_num else 1  # Use episode number as chapter ID
            paragraph_id = i + 1  # Paragraph ID within episode
            sentence_id = 1  # Single sentence per segment
            
            utterance_id = f"{speaker_id:03d}_{chapter_id:03d}_{paragraph_id:06d}_{sentence_id:06d}"
            
            # Create LibriTTS directory structure: {speaker_id}/{chapter_id}/
            speaker_dir = os.path.join(output_dir, f"{speaker_id:03d}")
            chapter_dir = os.path.join(speaker_dir, f"{chapter_id:03d}")
            os.makedirs(chapter_dir, exist_ok=True)
            
            # Save audio segment
            audio_filename = f"{utterance_id}.wav"
            audio_path_full = os.path.join(chapter_dir, audio_filename)
            sf.write(audio_path_full, segment_audio, sr)
            
            # Extract corresponding text from subtitles
            segment_text = ""
            if subtitles:
                for sub_time, sub_text in subtitles:
                    if start <= sub_time <= end:
                        segment_text += sub_text + " "
            
            segment_text = segment_text.strip()
            
            # Save normalized text file
            text_filename = f"{utterance_id}.normalized.txt"
            text_path = os.path.join(chapter_dir, text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(segment_text)
            
            # Add to TSV data for this speaker/chapter directory
            dir_key = f"{speaker_id:03d}/{chapter_id:03d}"
            if dir_key not in tsv_data_by_dir:
                tsv_data_by_dir[dir_key] = []
            
            tsv_data_by_dir[dir_key].append({
                'utterance_id': utterance_id,
                'original_text': segment_text,
                'normalized_text': segment_text
            })
            
            print(f"Saved: {utterance_id} - Speaker: {speaker_id} - Text: {len(segment_text)} chars")
        else:
            print(f"Skipped empty segment: speaker_{speaker}_{i:03d} ({start:.2f}s-{end:.2f}s)")
    
    # Save TSV files for each speaker/chapter directory
    for dir_key, tsv_data in tsv_data_by_dir.items():
        speaker_dir = os.path.join(output_dir, dir_key)
        tsv_path = os.path.join(speaker_dir, f"{dir_key.replace('/', '_')}.trans.tsv")
        
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("utterance_id\toriginal_text\tnormalized_text\n")
            for entry in tsv_data:
                f.write(f"{entry['utterance_id']}\t{entry['original_text']}\t{entry['normalized_text']}\n")
        
        print(f"📊 TSV file saved: {tsv_path}")
    
    print(f"📋 Speaker mapping: {speaker_mapping}")
    
    return tsv_data_by_dir, speaker_mapping


def main():
    parser = argparse.ArgumentParser(description="Pyannote speaker diarization and audio segmentation")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("subtitle_file", help="Path to the subtitle file")
    parser.add_argument("--output_dir", default="segmented_audio", help="Output directory for segmented files")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Minimum segment duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Maximum segment duration in seconds")
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1", help="Pyannote model to use")
    parser.add_argument("--episode_num", type=int, help="Episode number for filename prefix")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.subtitle_file):
        print(f"Error: Subtitle file not found: {args.subtitle_file}")
        sys.exit(1)
    
    print("Starting speaker diarization and segmentation process...")
    
    # Parse subtitle file
    print("Parsing subtitle file...")
    subtitles = parse_subtitle_file(args.subtitle_file)
    print(f"Found {len(subtitles)} subtitle entries")
    
    # Perform speaker diarization
    print("Performing speaker diarization...")
    diarization = perform_speaker_diarization(args.audio_file, args.model)
    
    # Convert diarization to segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    
    print(f"Found {len(segments)} speaker segments")
    
    # Sort segments by start time
    segments.sort(key=lambda x: x[0])
    
    # Merge consecutive segments from same speaker
    print("Merging consecutive same-speaker segments...")
    merged_segments = merge_consecutive_segments(segments, args.max_duration)
    print(f"After merging: {len(merged_segments)} segments")
    
    # Filter segments by duration
    print("Filtering segments by duration...")
    valid_segments, invalid_segments = filter_segments_by_duration(
        merged_segments, args.min_duration, args.max_duration
    )
    
    print(f"Valid segments: {len(valid_segments)}")
    print(f"Invalid segments (too short/long): {len(invalid_segments)}")
    
    # Segment and save audio
    print("Segmenting and saving audio files...")
    segment_audio(args.audio_file, valid_segments, args.output_dir, subtitles, args.episode_num)
    
    # Save invalid segments info
    if invalid_segments:
        invalid_info_path = os.path.join(args.output_dir, "invalid_segments.txt")
        with open(invalid_info_path, 'w', encoding='utf-8') as f:
            f.write("Invalid segments (too short or too long):\n")
            for start, end, speaker in invalid_segments:
                duration = end - start
                f.write(f"Speaker {speaker}: {start:.2f}s-{end:.2f}s ({duration:.2f}s)\n")
        print(f"Invalid segments info saved to: {invalid_info_path}")
    
    print("Process completed!")


if __name__ == "__main__":
    main()