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

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and test sets")
    parser.add_argument("data_dir", help="Path to the dataset directory")
    parser.add_argument("--output_dir", default="split_dataset", help="Output directory for split dataset")
    parser.add_argument("--method", choices=["speaker", "files", "episode"], default="files", 
                       help="Split method: speaker, files, or episode")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio (for speaker/files methods)")
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
    
    if args.method == "files":
        split_by_files(args.data_dir, args.output_dir, args.test_ratio)
    
    print("✅ Dataset split completed!")

if __name__ == "__main__":
    main()