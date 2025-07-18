#!/usr/bin/env python3
"""
Split dataset into train and test sets
"""

import os
import shutil
import random
from pathlib import Path
import argparse
from typing import List, Dict, Tuple


def get_all_speakers(data_dir: str) -> List[str]:
    """Get all speaker IDs from the dataset."""
    speakers = []
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)) and item.isdigit():
            speakers.append(item)
    return sorted(speakers)


def get_speaker_files(data_dir: str, speaker_id: str) -> List[Tuple[str, str]]:
    """Get all audio files for a specific speaker."""
    files = []
    speaker_dir = os.path.join(data_dir, speaker_id)
    
    for chapter in os.listdir(speaker_dir):
        chapter_dir = os.path.join(speaker_dir, chapter)
        if os.path.isdir(chapter_dir):
            for file in os.listdir(chapter_dir):
                if file.endswith('.wav'):
                    files.append((chapter, file))
    
    return files


def split_by_speaker(data_dir: str, output_dir: str, test_ratio: float = 0.2):
    """Split dataset by speaker (speakers are either in train or test)."""
    speakers = get_all_speakers(data_dir)
    random.shuffle(speakers)
    
    test_count = int(len(speakers) * test_ratio)
    test_speakers = speakers[:test_count]
    train_speakers = speakers[test_count:]
    
    print(f"📊 Total speakers: {len(speakers)}")
    print(f"📊 Train speakers: {len(train_speakers)} - {train_speakers}")
    print(f"📊 Test speakers: {len(test_speakers)} - {test_speakers}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files
    for speaker in train_speakers:
        src = os.path.join(data_dir, speaker)
        dst = os.path.join(train_dir, speaker)
        shutil.copytree(src, dst)
    
    for speaker in test_speakers:
        src = os.path.join(data_dir, speaker)
        dst = os.path.join(test_dir, speaker)
        shutil.copytree(src, dst)


def split_by_files(data_dir: str, output_dir: str, test_ratio: float = 0.2):
    """Split dataset by files (each speaker has files in both train and test)."""
    speakers = get_all_speakers(data_dir)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    for speaker in speakers:
        files = get_speaker_files(data_dir, speaker)
        random.shuffle(files)
        
        test_file_count = int(len(files) * test_ratio)
        test_files = files[:test_file_count]
        train_files = files[test_file_count:]
        
        print(f"📊 Speaker {speaker}: {len(train_files)} train, {len(test_files)} test")
        
        # Copy train files
        for chapter, filename in train_files:
            src_dir = os.path.join(data_dir, speaker, chapter)
            dst_dir = os.path.join(train_dir, speaker, chapter)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy audio file
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
            
            # Copy text file
            txt_filename = filename.replace('.wav', '.normalized.txt')
            if os.path.exists(os.path.join(src_dir, txt_filename)):
                shutil.copy2(os.path.join(src_dir, txt_filename), os.path.join(dst_dir, txt_filename))
            
            # Copy TSV file (if exists)
            tsv_filename = f"{speaker}_{chapter}.trans.tsv"
            if os.path.exists(os.path.join(src_dir, tsv_filename)):
                shutil.copy2(os.path.join(src_dir, tsv_filename), os.path.join(dst_dir, tsv_filename))
            
            train_count += 1
        
        # Copy test files
        for chapter, filename in test_files:
            src_dir = os.path.join(data_dir, speaker, chapter)
            dst_dir = os.path.join(test_dir, speaker, chapter)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy audio file
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
            
            # Copy text file
            txt_filename = filename.replace('.wav', '.normalized.txt')
            if os.path.exists(os.path.join(src_dir, txt_filename)):
                shutil.copy2(os.path.join(src_dir, txt_filename), os.path.join(dst_dir, txt_filename))
            
            # Copy TSV file (if exists)
            tsv_filename = f"{speaker}_{chapter}.trans.tsv"
            if os.path.exists(os.path.join(src_dir, tsv_filename)):
                shutil.copy2(os.path.join(src_dir, tsv_filename), os.path.join(dst_dir, tsv_filename))
            
            test_count += 1
    
    print(f"📊 Total: {train_count} train files, {test_count} test files")


def split_by_episode(data_dir: str, output_dir: str, test_episodes: List[int]):
    """Split dataset by episode (chapters)."""
    speakers = get_all_speakers(data_dir)
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    for speaker in speakers:
        speaker_dir = os.path.join(data_dir, speaker)
        
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if os.path.isdir(chapter_dir):
                chapter_num = int(chapter)
                
                if chapter_num in test_episodes:
                    # Copy to test
                    dst_dir = os.path.join(test_dir, speaker, chapter)
                    shutil.copytree(chapter_dir, dst_dir)
                    test_count += len([f for f in os.listdir(dst_dir) if f.endswith('.wav')])
                else:
                    # Copy to train
                    dst_dir = os.path.join(train_dir, speaker, chapter)
                    shutil.copytree(chapter_dir, dst_dir)
                    train_count += len([f for f in os.listdir(dst_dir) if f.endswith('.wav')])
    
    print(f"📊 Test episodes: {test_episodes}")
    print(f"📊 Total: {train_count} train files, {test_count} test files")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("data_dir", help="Path to the dataset directory")
    parser.add_argument("--output_dir", default="split_dataset", help="Output directory for split dataset")
    parser.add_argument("--method", choices=["speaker", "files", "episode"], default="files", 
                       help="Split method: speaker, files, or episode")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (for speaker/files methods)")
    parser.add_argument("--test_episodes", nargs="+", type=int, help="Episode numbers for test set (for episode method)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    print(f"🔄 Splitting dataset using '{args.method}' method...")
    print(f"📁 Input directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    if args.method == "speaker":
        split_by_speaker(args.data_dir, args.output_dir, args.test_ratio)
    elif args.method == "files":
        split_by_files(args.data_dir, args.output_dir, args.test_ratio)
    elif args.method == "episode":
        if not args.test_episodes:
            print("❌ Please specify --test_episodes for episode method")
            return
        split_by_episode(args.data_dir, args.output_dir, args.test_episodes)
    
    print("✅ Dataset split completed!")


if __name__ == "__main__":
    main()