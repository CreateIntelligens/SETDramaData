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
            
            # Copy audio file (使用最基本的複製方式避免權限問題)
            with open(os.path.join(src_dir, filename), 'rb') as src, open(os.path.join(dst_dir, filename), 'wb') as dst:
                dst.write(src.read())
            
            # Copy text file
            txt_filename = filename.replace('.wav', '.normalized.txt')
            if os.path.exists(os.path.join(src_dir, txt_filename)):
                with open(os.path.join(src_dir, txt_filename), 'rb') as src, open(os.path.join(dst_dir, txt_filename), 'wb') as dst:
                    dst.write(src.read())
            
            train_count += 1
        
        # Copy test files  
        for chapter, filename in test_files:
            src_dir = os.path.join(data_dir, speaker, chapter)
            dst_dir = os.path.join(test_dir, speaker, chapter)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Copy audio file (使用最基本的複製方式避免權限問題)
            with open(os.path.join(src_dir, filename), 'rb') as src, open(os.path.join(dst_dir, filename), 'wb') as dst:
                dst.write(src.read())
            
            # Copy text file
            txt_filename = filename.replace('.wav', '.normalized.txt')
            if os.path.exists(os.path.join(src_dir, txt_filename)):
                with open(os.path.join(src_dir, txt_filename), 'rb') as src, open(os.path.join(dst_dir, txt_filename), 'wb') as dst:
                    dst.write(src.read())
            
            test_count += 1
    
    print(f"📊 Total: {train_count} train files, {test_count} test files")

def split_by_episode(processed_dir: str, split_dir: str, episode_num: str, test_ratio: float = 0.2):
    """Split a specific episode's data into train and test sets."""
    print(f"按集數切分 - 處理第 {episode_num} 集")
    
    # Create output directories
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    train_count = 0
    test_count = 0
    
    # Find all speakers that have data for this episode
    episode_speakers = []
    for speaker in os.listdir(processed_dir):
        if os.path.isdir(os.path.join(processed_dir, speaker)) and speaker.isdigit():
            episode_dir = os.path.join(processed_dir, speaker, episode_num)
            if os.path.exists(episode_dir) and os.path.isdir(episode_dir):
                files = [f for f in os.listdir(episode_dir) if f.endswith('.wav')]
                if files:  # Only include speakers with actual audio files
                    episode_speakers.append(speaker)
    
    if not episode_speakers:
        print(f"找不到第 {episode_num} 集的音檔資料")
        return
    
    print(f"第 {episode_num} 集包含 {len(episode_speakers)} 個語者: {', '.join(episode_speakers)}")
    
    # For each speaker in this episode, split their files
    for speaker in episode_speakers:
        episode_dir = os.path.join(processed_dir, speaker, episode_num)
        files = [f for f in os.listdir(episode_dir) if f.endswith('.wav')]
        
        if not files:
            continue
            
        # Shuffle files for this speaker
        random.shuffle(files)
        
        # Calculate split point
        split_point = int(len(files) * (1 - test_ratio))
        train_files = files[:split_point]
        test_files = files[split_point:]
        
        # Copy train files
        for filename in train_files:
            train_speaker_dir = os.path.join(train_dir, speaker)
            os.makedirs(train_speaker_dir, exist_ok=True)
            
            src_file = os.path.join(episode_dir, filename)
            dst_file = os.path.join(train_speaker_dir, filename)
            
            # Copy audio file
            with open(src_file, 'rb') as src, open(dst_file, 'wb') as dst:
                dst.write(src.read())
            
            # Copy text file if exists
            txt_filename = filename.replace('.wav', '.normalized.txt')
            src_txt = os.path.join(episode_dir, txt_filename)
            if os.path.exists(src_txt):
                dst_txt = os.path.join(train_speaker_dir, txt_filename)
                with open(src_txt, 'rb') as src, open(dst_txt, 'wb') as dst:
                    dst.write(src.read())
            
            train_count += 1
        
        # Copy test files  
        for filename in test_files:
            test_speaker_dir = os.path.join(test_dir, speaker)
            os.makedirs(test_speaker_dir, exist_ok=True)
            
            src_file = os.path.join(episode_dir, filename)
            dst_file = os.path.join(test_speaker_dir, filename)
            
            # Copy audio file
            with open(src_file, 'rb') as src, open(dst_file, 'wb') as dst:
                dst.write(src.read())
            
            # Copy text file if exists
            txt_filename = filename.replace('.wav', '.normalized.txt')
            src_txt = os.path.join(episode_dir, txt_filename)
            if os.path.exists(src_txt):
                dst_txt = os.path.join(test_speaker_dir, txt_filename)
                with open(src_txt, 'rb') as src, open(dst_txt, 'wb') as dst:
                    dst.write(src.read())
            
            test_count += 1
    
    print(f"第 {episode_num} 集切分完成: {train_count} train files, {test_count} test files")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("--processed_dir", default="data/output", help="Path to the processed dataset directory")
    parser.add_argument("--split_dir", default="data/split_dataset", help="Output directory for split dataset")
    parser.add_argument("--method", choices=["speaker", "files", "episode"], default="files", 
                       help="Split method: speaker, files, or episode")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (for speaker/files methods)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", help="Specific episodes to split (e.g., '1,2,3')")
    parser.add_argument("--episode_num", help="Single episode to split (for episode method)")
    parser.add_argument("--auto_yes", action="store_true", help="Auto confirm all prompts")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if not os.path.exists(args.processed_dir):
        print(f"❌ Processed directory not found: {args.processed_dir}")
        return
    
    print(f"切分資料集使用 '{args.method}' 方法...")
    print(f"輸入目錄: {args.processed_dir}")
    print(f"輸出目錄: {args.split_dir}")
    
    if args.method == "files":
        split_by_files(args.processed_dir, args.split_dir, args.test_ratio)
    elif args.method == "episode":
        if not args.episode_num:
            print("Episode method requires --episode_num parameter")
            return
        split_by_episode(args.processed_dir, args.split_dir, args.episode_num, args.test_ratio)
    else:
        print(f"Method '{args.method}' not implemented yet")
        return
    
    print("Dataset split completed!")

if __name__ == "__main__":
    main()