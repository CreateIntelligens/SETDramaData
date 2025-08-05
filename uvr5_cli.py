#!/usr/bin/env python3
"""
UVR5 專用命令行工具
快速處理單檔、多檔或目錄，支援萬用字元匹配

使用範例：
  python uvr5_cli.py input.wav                    # 單檔處理
  python uvr5_cli.py data/audio/                  # 整個目錄
  python uvr5_cli.py "back*.wav"                  # 萬用字元匹配
  python uvr5_cli.py data/test/ --pattern "*.mp3" # 指定格式
  python uvr5_cli.py input.wav --backup           # 備份原檔
  python uvr5_cli.py data/ --threads 2            # 多執行緒

Author: Breeze ASR ETL Pipeline
Version: 1.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List
import glob

# 導入 UVR5 處理器
try:
    from src.uvr5_processor import UVR5Processor, ThreadedUVR5Processor
except ImportError:
    try:
        from uvr5_processor import UVR5Processor, ThreadedUVR5Processor
    except ImportError:
        print("❌ 無法導入 UVR5Processor，請確認 src/uvr5_processor.py 存在")
        sys.exit(1)


def find_audio_files(pattern: str) -> List[Path]:
    """
    根據模式尋找音檔
    
    Args:
        pattern: 檔案路徑、目錄路徑或萬用字元模式
        
    Returns:
        List[Path]: 找到的音檔列表
    """
    pattern = str(pattern).strip()
    audio_files = []
    
    # 如果是具體檔案路徑
    if Path(pattern).is_file():
        audio_files = [Path(pattern)]
        print(f"📄 找到單一檔案: {pattern}")
    
    # 如果是目錄路徑
    elif Path(pattern).is_dir():
        directory = Path(pattern)
        audio_files = list(directory.rglob("*.wav")) + list(directory.rglob("*.mp3")) + list(directory.rglob("*.flac"))
        print(f"📁 掃描目錄: {pattern}")
        print(f"🎵 找到 {len(audio_files)} 個音檔")
    
    # 如果包含萬用字元
    elif '*' in pattern or '?' in pattern or '[' in pattern:
        matched_files = glob.glob(pattern, recursive=True)
        audio_files = [Path(f) for f in matched_files if Path(f).is_file()]
        print(f"🔍 萬用字元匹配: {pattern}")
        print(f"🎵 找到 {len(audio_files)} 個檔案")
    
    # 嘗試作為檔案路徑（可能不存在）
    else:
        file_path = Path(pattern)
        if file_path.exists():
            audio_files = [file_path]
        else:
            print(f"❌ 找不到檔案或目錄: {pattern}")
    
    return audio_files


def main():
    parser = argparse.ArgumentParser(
        description="UVR5 音頻人聲分離工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s input.wav                           # 處理單個檔案
  %(prog)s data/audio/                         # 處理整個目錄
  %(prog)s "backup_*.wav"                      # 萬用字元匹配
  %(prog)s "data/**/*.mp3"                     # 遞迴匹配所有mp3
  %(prog)s data/ --pattern "*.flac"            # 指定檔案格式
  %(prog)s input.wav --backup                  # 備份原始檔案
  %(prog)s data/ --threads 3                   # 多執行緒處理
  %(prog)s input.wav --min-duration 5         # 自訂最小長度
        """
    )
    
    parser.add_argument(
        'input',
        help='輸入檔案、目錄或萬用字元模式 (如: "backup_*.wav")'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        default='*.wav',
        help='檔案匹配模式 (預設: *.wav)'
    )
    
    parser.add_argument(
        '--backup', '-b',
        action='store_true',
        help='備份原始檔案為 .bak'
    )
    
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=1,
        help='執行緒數量 (預設: 1，建議 1-3)'
    )
    
    parser.add_argument(
        '--min-duration',
        type=float,
        help='最小音檔長度（秒），短於此值會補零'
    )
    
    parser.add_argument(
        '--target-duration',
        type=float,
        help='補零目標長度（秒）'
    )
    
    parser.add_argument(
        '--model-path',
        default='models/uvr5',
        help='UVR5 模型目錄路徑'
    )
    
    parser.add_argument(
        '--vocal-model',
        default='model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        help='UVR5 人聲模型檔名'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='處理裝置 (預設: auto)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只顯示會處理的檔案，不實際執行'
    )
    
    args = parser.parse_args()
    
    print("🎯 UVR5 專用處理工具")
    print("=" * 50)
    
    # 尋找要處理的檔案
    if Path(args.input).is_dir():
        # 目錄模式：使用 pattern 參數
        audio_files = list(Path(args.input).rglob(args.pattern))
    else:
        # 檔案或萬用字元模式
        audio_files = find_audio_files(args.input)
    
    if not audio_files:
        print("❌ 沒有找到任何音檔")
        return 1
    
    # 顯示找到的檔案
    print(f"\n📋 找到 {len(audio_files)} 個音檔:")
    for i, file_path in enumerate(audio_files[:10], 1):  # 只顯示前10個
        print(f"  {i:2d}. {file_path}")
    
    if len(audio_files) > 10:
        print(f"  ... 還有 {len(audio_files) - 10} 個檔案")
    
    # Dry run 模式
    if args.dry_run:
        print(f"\n✅ Dry run 完成，共 {len(audio_files)} 個檔案會被處理")
        return 0
    
    # 確認處理
    if len(audio_files) > 5:
        response = input(f"\n❓ 確定要處理 {len(audio_files)} 個檔案嗎？(y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 使用者取消")
            return 0
    
    # 初始化處理器參數
    processor_kwargs = {
        'model_path': args.model_path,
        'vocal_model': args.vocal_model,
        'device': args.device,
    }
    
    if args.min_duration is not None:
        processor_kwargs['min_duration'] = args.min_duration
    
    if args.target_duration is not None:
        processor_kwargs['target_duration'] = args.target_duration
    
    try:
        # 選擇處理器類型
        if args.threads > 1:
            print(f"🚀 使用多執行緒處理器 (執行緒數: {args.threads})")
            processor = ThreadedUVR5Processor(
                max_workers=args.threads,
                **processor_kwargs
            )
        else:
            print("🔄 使用單執行緒處理器")
            processor = UVR5Processor(**processor_kwargs)
        
        # 顯示處理器資訊
        model_info = processor.get_model_info()
        print(f"\n📋 處理器資訊:")
        print(f"  模型: {model_info['vocal_model']}")
        print(f"  裝置: {model_info['device']}")
        print(f"  最小長度: {model_info['min_duration']}s")
        print(f"  目標長度: {model_info['target_duration']}s")
        
        if not model_info['model_exists']:
            print(f"\n❌ UVR5 模型檔案不存在: {args.model_path}/{args.vocal_model}")
            return 1
        
        print(f"\n🎵 開始處理...")
        start_time = time.time()
        
        # 處理檔案
        if len(audio_files) == 1:
            # 單檔處理
            result = processor.enhance_audio(
                str(audio_files[0]),
                backup_original=args.backup
            )
            
            if result['success']:
                processing_time = result['processing_time']
                print(f"✅ 處理完成: {audio_files[0].name}")
                print(f"⏱️  處理時間: {processing_time:.2f} 秒")
                if result.get('already_processed'):
                    print("ℹ️  檔案已處理過（跳過）")
            else:
                print(f"❌ 處理失敗: {result.get('error', 'Unknown error')}")
                return 1
        
        else:
            # 批量處理 - 創建臨時目錄來使用 batch_enhance
            if len(set(f.parent for f in audio_files)) == 1:
                # 所有檔案在同一目錄
                input_dir = audio_files[0].parent
                result = processor.batch_enhance(
                    str(input_dir),
                    pattern=args.pattern,
                    backup_original=args.backup
                )
            else:
                # 檔案分散在不同目錄，逐個處理
                success_count = 0
                failed_count = 0
                
                for audio_file in audio_files:
                    print(f"🎵 處理: {audio_file.name}")
                    result = processor.enhance_audio(
                        str(audio_file),
                        backup_original=args.backup
                    )
                    
                    if result['success']:
                        success_count += 1
                        if not result.get('already_processed'):
                            print(f"  ✅ 完成 ({result['processing_time']:.2f}s)")
                        else:
                            print(f"  ⏭️  已處理過")
                    else:
                        failed_count += 1
                        print(f"  ❌ 失敗: {result.get('error', 'Unknown error')}")
                
                print(f"\n📊 處理結果:")
                print(f"  成功: {success_count} 檔案")
                print(f"  失敗: {failed_count} 檔案")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  總處理時間: {total_time:.2f} 秒")
        print("🎉 全部完成！")
        
        return 0
        
    except Exception as e:
        print(f"❌ 處理過程發生錯誤: {e}")
        return 1
    
    finally:
        if 'processor' in locals():
            processor.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)