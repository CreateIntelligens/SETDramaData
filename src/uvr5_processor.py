#!/usr/bin/env python3
"""
UVR5 音頻增強處理器
針對切分後的短音檔進行音質改善

主要功能：
- 單檔音頻增強處理
- 批量目錄處理
- 切分資料集增強
- 記憶體友善設計

Author: Breeze ASR ETL Pipeline
Version: 1.0
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torchaudio
except ImportError:
    print("❌ 請安裝 PyTorch: pip install torch torchaudio")
    sys.exit(1)

try:
    from audio_separator.separator import Separator
except ImportError:
    print("❌ 請安裝 audio-separator: pip install 'audio-separator[gpu]'")
    sys.exit(1)

from tqdm import tqdm
import psutil


class UVR5Processor:
    """UVR5 音頻增強處理器 - 專門處理切分後的短音檔
    
    設計原則：
    - 記憶體友善：單檔處理避免記憶體溢出
    - 錯誤隔離：單檔失敗不影響其他檔案
    - 進度透明：清楚顯示處理進度和狀態
    - 配置靈活：透過參數控制處理行為
    """
    
    def __init__(self, 
                 model_path: str = "models/uvr5",
                 vocal_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                 device: str = "auto",
                 batch_size: int = 1):
        """
        初始化 UVR5 處理器
        
        Args:
            model_path: UVR5 模型目錄路徑
            vocal_model: 人聲分離模型檔名
            device: 處理設備 ('cuda', 'cpu', 'auto')
            batch_size: 批次大小 (推薦為1以節省記憶體)
        """
        self.model_path = Path(model_path)
        self.vocal_model = vocal_model
        self.batch_size = batch_size
        
        # 設定 logging
        self.setup_logging()
        
        # 設定裝置
        self.device = self._setup_device(device)
        
        # 初始化分離器
        self.separator = None
        self._setup_separator()
        
        # 統計資訊
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': []
        }
    
    def setup_logging(self):
        """設定 logging 系統"""
        log_file = Path.cwd() / 'uvr5_processor.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self, device: str) -> str:
        """設定處理裝置
        
        Args:
            device (str): 裝置類型 ('auto', 'cuda', 'cpu')
            
        Returns:
            str: 實際使用的裝置類型
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"🎮 使用 GPU: {gpu_name}")
            else:
                device = "cpu"
                self.logger.info("🖥️  使用 CPU 模式")
        
        return device
    
    def _setup_separator(self):
        """初始化 UVR5 分離器"""
        try:
            # 檢查模型檔案
            model_file = self.model_path / self.vocal_model
            if not model_file.exists():
                raise FileNotFoundError(f"UVR5 模型檔案不存在: {model_file}")
            
            # 創建分離器配置
            separator_config = {
                'log_level': logging.WARNING,  # 減少 UVR5 日誌輸出
                'output_format': 'WAV',
                'normalization_threshold': 0.9,
                'sample_rate': 44100,
                'use_autocast': self.device == 'cuda',
                'mdx_params': {
                    "segment_size": 1024,  # 較小的分段大小適合短音檔
                    "overlap": 0.25,       # 適中的重疊率
                    "batch_size": self.batch_size,
                    "enable_denoise": True
                }
            }
            
            self.separator = Separator(**separator_config)
            self.logger.info(f"✅ UVR5 分離器初始化成功")
            
        except Exception as e:
            self.logger.error(f"❌ UVR5 分離器初始化失敗: {e}")
            raise
    
    def enhance_audio(self, input_path: str, output_path: Optional[str] = None, 
                     backup_original: bool = False) -> Dict:
        """
        對單個音檔進行 UVR5 增強處理
        
        Args:
            input_path: 輸入音檔路徑
            output_path: 輸出路徑 (None = 原地替換)
            backup_original: 是否備份原始檔案
            
        Returns:
            Dict: 處理結果
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"輸入檔案不存在: {input_path}")
        
        # 決定輸出路徑
        if output_path is None:
            output_path = input_path  # 原地替換
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'success': False,
            'processing_time': 0,
            'memory_usage_mb': 0,
            'error': None,
            'enhanced': False
        }
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / (1024**2)
        
        try:
            # 載入模型 (每次處理時載入以節省記憶體)
            if self.separator.model is None:
                self.separator.load_model(model_filename=self.vocal_model)
            
            # 備份原始檔案
            if backup_original and output_path == input_path:
                backup_path = input_path.with_suffix(f'.backup{input_path.suffix}')
                input_path.rename(backup_path)
                input_path = backup_path
                result['backup_file'] = str(backup_path)
            
            # 臨時輸出目錄
            temp_output_dir = output_path.parent / f"temp_uvr5_{int(time.time())}"
            temp_output_dir.mkdir(exist_ok=True)
            
            try:
                # 執行分離
                self.separator.output_dir = str(temp_output_dir)
                output_files = self.separator.separate(str(input_path))
                
                # 尋找人聲檔案
                vocals_file = None
                for file_path in output_files:
                    if 'vocals' in Path(file_path).name.lower():
                        vocals_file = file_path
                        break
                
                if vocals_file and Path(vocals_file).exists():
                    # 移動人聲檔案到目標位置
                    import shutil
                    shutil.move(vocals_file, output_path)
                    result['enhanced'] = True
                    self.logger.info(f"✅ 音頻增強完成: {input_path.name}")
                else:
                    raise RuntimeError("人聲檔案生成失敗")
                
            finally:
                # 清理臨時目錄
                import shutil
                if temp_output_dir.exists():
                    shutil.rmtree(temp_output_dir)
            
            # 計算處理時間和記憶體使用
            processing_time = time.time() - start_time
            current_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_usage = current_memory - initial_memory
            
            result.update({
                'success': True,
                'processing_time': processing_time,
                'memory_usage_mb': memory_usage
            })
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            self.logger.error(f"❌ 音頻增強失敗 {input_path.name}: {e}")
            
            # 恢復原始檔案 (如果有備份)
            if backup_original and 'backup_file' in result:
                backup_path = Path(result['backup_file'])
                if backup_path.exists():
                    backup_path.rename(output_path)
        
        finally:
            # 清理記憶體
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        return result
    
    def batch_enhance(self, input_dir: str, pattern: str = "*.wav",
                     backup_original: bool = False) -> Dict:
        """
        批量處理目錄下的音檔
        
        Args:
            input_dir: 輸入目錄
            pattern: 檔案匹配模式
            backup_original: 是否備份原始檔案
            
        Returns:
            Dict: 批量處理結果
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"輸入目錄不存在: {input_dir}")
        
        # 尋找音檔
        audio_files = list(input_dir.glob(pattern))
        if not audio_files:
            self.logger.warning(f"在 {input_dir} 中未找到匹配 {pattern} 的音檔")
            return {'success': False, 'error': 'No audio files found'}
        
        self.logger.info(f"📁 找到 {len(audio_files)} 個音檔進行處理")
        
        # 重置統計
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': []
        }
        
        start_time = time.time()
        
        # 批量處理
        for audio_file in tqdm(audio_files, desc="🎵 UVR5 音頻增強"):
            try:
                result = self.enhance_audio(str(audio_file), backup_original=backup_original)
                
                if result['success']:
                    self.stats['processed_files'] += 1
                else:
                    self.stats['failed_files'] += 1
                    self.stats['failed_list'].append({
                        'file': str(audio_file),
                        'error': result.get('error', 'Unknown error')
                    })
                
            except Exception as e:
                self.stats['failed_files'] += 1
                self.stats['failed_list'].append({
                    'file': str(audio_file),
                    'error': str(e)
                })
                self.logger.error(f"❌ 處理失敗 {audio_file.name}: {e}")
        
        self.stats['total_time'] = time.time() - start_time
        
        # 生成報告
        self._generate_batch_report()
        
        return {
            'success': True,
            'stats': self.stats,
            'total_files': len(audio_files),
            'processed_files': self.stats['processed_files'],
            'failed_files': self.stats['failed_files']
        }
    
    def enhance_split_dataset(self, split_dir: str, backup_original: bool = False) -> Dict:
        """
        對切分後的訓練/測試集進行增強處理
        
        Args:
            split_dir: split_dataset 目錄路徑
            backup_original: 是否備份原始檔案
            
        Returns:
            Dict: 處理結果
        """
        split_dir = Path(split_dir)
        if not split_dir.exists():
            raise FileNotFoundError(f"切分資料集目錄不存在: {split_dir}")
        
        results = {}
        
        # 處理 train 和 test 目錄
        for subset in ['train', 'test']:
            subset_dir = split_dir / subset
            if not subset_dir.exists():
                self.logger.warning(f"⚠️  {subset} 目錄不存在: {subset_dir}")
                continue
            
            self.logger.info(f"📊 開始處理 {subset} 資料集...")
            
            # 處理每個說話人目錄
            speaker_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
            subset_results = []
            
            for speaker_dir in speaker_dirs:
                self.logger.info(f"👤 處理說話人: {speaker_dir.name}")
                
                try:
                    result = self.batch_enhance(
                        str(speaker_dir), 
                        pattern="*.wav",
                        backup_original=backup_original
                    )
                    result['speaker'] = speaker_dir.name
                    subset_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"❌ 說話人 {speaker_dir.name} 處理失敗: {e}")
                    subset_results.append({
                        'speaker': speaker_dir.name,
                        'success': False,
                        'error': str(e)
                    })
            
            results[subset] = subset_results
        
        # 生成整體報告
        self._generate_split_dataset_report(results)
        
        return {
            'success': True,
            'results': results
        }
    
    def _generate_batch_report(self):
        """生成批量處理報告"""
        print("\n" + "="*60)
        print("🎵 UVR5 音頻增強批量處理報告")
        print("="*60)
        print(f"📊 處理統計:")
        print(f"  成功處理: {self.stats['processed_files']} 檔案")
        print(f"  處理失敗: {self.stats['failed_files']} 檔案")
        print(f"  總處理時間: {self.stats['total_time']:.2f} 秒")
        
        if self.stats['processed_files'] > 0:
            avg_time = self.stats['total_time'] / self.stats['processed_files']
            print(f"  平均處理時間: {avg_time:.2f} 秒/檔")
        
        if self.stats['failed_files'] > 0:
            print(f"\n❌ 失敗檔案清單:")
            for failed in self.stats['failed_list']:
                print(f"  • {failed['file']}: {failed['error']}")
    
    def _generate_split_dataset_report(self, results: Dict):
        """生成切分資料集處理報告"""
        print("\n" + "="*60)
        print("📊 切分資料集 UVR5 增強報告")
        print("="*60)
        
        for subset, subset_results in results.items():
            print(f"\n📁 {subset.upper()} 資料集:")
            
            total_speakers = len(subset_results)
            successful_speakers = sum(1 for r in subset_results if r.get('success', False))
            total_files = sum(r.get('processed_files', 0) for r in subset_results if r.get('success', False))
            
            print(f"  說話人總數: {total_speakers}")
            print(f"  成功處理說話人: {successful_speakers}")
            print(f"  總處理檔案數: {total_files}")
            
            # 顯示失敗的說話人
            failed_speakers = [r for r in subset_results if not r.get('success', False)]
            if failed_speakers:
                print(f"  ❌ 失敗說話人:")
                for failed in failed_speakers:
                    print(f"    • {failed['speaker']}: {failed.get('error', 'Unknown error')}")
    
    def get_model_info(self) -> Dict:
        """獲取模型資訊"""
        model_file = self.model_path / self.vocal_model
        
        return {
            'model_path': str(self.model_path),
            'vocal_model': self.vocal_model,
            'model_exists': model_file.exists(),
            'model_size_mb': model_file.stat().st_size / (1024**2) if model_file.exists() else 0,
            'device': self.device,
            'batch_size': self.batch_size
        }
    
    def cleanup(self):
        """清理資源"""
        if self.separator:
            del self.separator
            self.separator = None
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        gc.collect()
        self.logger.info("🧹 UVR5 處理器資源清理完成")


def main():
    """測試 UVR5 處理器"""
    print("🎯 UVR5 處理器測試")
    print("=" * 50)
    
    try:
        processor = UVR5Processor()
        
        # 顯示模型資訊
        model_info = processor.get_model_info()
        print("📋 模型資訊:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        if not model_info['model_exists']:
            print("\n❌ UVR5 模型檔案不存在")
            print("請將模型檔案放置到 models/uvr5/ 目錄")
            return False
        
        print("\n✅ UVR5 處理器初始化成功")
        print("💡 使用方式:")
        print("  processor.enhance_audio('input.wav')  # 單檔處理")
        print("  processor.batch_enhance('audio_dir')  # 批量處理")
        print("  processor.enhance_split_dataset('data/split_dataset')  # 處理切分資料集")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False
    
    finally:
        if 'processor' in locals():
            processor.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)