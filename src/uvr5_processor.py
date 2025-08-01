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
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

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
                 batch_size: int = 1,
                 min_duration: float = None,
                 target_duration: float = None,
                 processing_timeout: int = None):
        """
        初始化 UVR5 處理器
        """
        self.model_path = Path(model_path)
        self.vocal_model = vocal_model
        self.batch_size = batch_size
        
        import os
        self.min_duration = min_duration if min_duration is not None else float(os.getenv('UVR5_MIN_DURATION', '10.0'))
        self.target_duration = target_duration if target_duration is not None else float(os.getenv('UVR5_TARGET_DURATION', '15.0'))
        self.processing_timeout = processing_timeout if processing_timeout is not None else int(os.getenv('UVR5_PROCESSING_TIMEOUT', '300'))
        
        self.setup_logging()
        
        # --- 建立並清理專用的暫存目錄 ---
        self.temp_dir = Path.cwd() / "data" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_temp_dir(initial_cleanup=True)

        self.device = self._setup_device(device)
        self.separator = None
        self._setup_separator()
        
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': []
        }

    def _cleanup_temp_dir(self, initial_cleanup: bool = False):
        """清理暫存目錄"""
        if initial_cleanup:
            self.logger.info(f"🧹 正在清理舊的暫存檔案於: {self.temp_dir}")
        
        for item in self.temp_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                self.logger.warning(f"⚠️ 清理暫存項目 {item} 失敗: {e}")

    def setup_logging(self):
        """設定 logging 系統"""
        log_file = Path.cwd() / 'uvr5_processor.log'
        
        logging.basicConfig(
            level=logging.DEBUG,
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
            
            # 創建分離器 - 針對短音檔優化
            self.separator = Separator(
                log_level=logging.WARNING,  # 減少 UVR5 日誌輸出
                model_file_dir=str(self.model_path),
                output_dir=str(self.temp_dir),  # 預設使用專用暫存目錄
                output_format='WAV',
                # 針對短音檔調整 MDX 參數
                mdx_params={
                    'segment_size': 128,  # 較小的分段適合短音檔
                    'overlap': 0.25,
                    'batch_size': 1,  # 確保穩定性
                    'hop_length': 512   # 較小的跳躍長度
                }
            )
            
            # 載入指定的模型
            self.separator.load_model(self.vocal_model)
            self.logger.info(f"✅ UVR5 分離器初始化成功")
            
        except Exception as e:
            self.logger.error(f"❌ UVR5 分離器初始化失敗: {e}")
            raise
    
    def get_audio_duration(self, audio_path: str) -> float:
        """獲取音頻檔案長度（秒）
        
        Args:
            audio_path: 音頻檔案路徑
            
        Returns:
            float: 音頻長度（秒）
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            return duration
        except Exception as e:
            self.logger.warning(f"⚠️  無法獲取音頻長度 {audio_path}: {e}")
            return 0.0
    
    def pad_audio_for_uvr5(self, input_path: str) -> Optional[str]:
        """為短音頻檔案進行補零預處理，並統一音頻格式
        
        Args:
            input_path: 輸入音頻檔案路徑
            
        Returns:
            Optional[str]: 預處理後的臨時檔案路徑，如果不需要預處理則返回 None
        """
        try:
            # 獲取音頻長度
            duration = self.get_audio_duration(input_path)
            
            if duration <= 0:
                self.logger.warning(f"⚠️  無效的音頻檔案: {input_path}")
                return None
            
            # 載入音頻
            waveform, sample_rate = torchaudio.load(input_path)
            
            # 檢查是否需要格式標準化
            needs_format_fix = False
            target_sample_rate = 44100  # UVR5 偏好的採樣率
            
            # 轉換為單聲道（如果是立體聲）
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                needs_format_fix = True
                self.logger.info(f"🔄 轉換立體聲為單聲道")
            
            # 重採樣到目標採樣率
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = target_sample_rate
                needs_format_fix = True
                self.logger.info(f"🔄 重採樣: {sample_rate}Hz → {target_sample_rate}Hz")
            
            # 檢查是否需要補零
            needs_padding = duration < self.min_duration
            
            if needs_padding:
                self.logger.info(f"📏 音頻長度 {duration:.2f}s < {self.min_duration}s，執行補零預處理...")
                
                # 計算需要的總樣本數
                target_samples = int(self.target_duration * sample_rate)
                current_samples = waveform.shape[1]
                
                if current_samples < target_samples:
                    # 計算需要補零的樣本數
                    padding_samples = target_samples - current_samples
                    
                    # 前後補零（平均分配）
                    padding_before = padding_samples // 2
                    padding_after = padding_samples - padding_before
                    
                    # 創建補零後的音頻
                    waveform = torch.nn.functional.pad(waveform, 
                                                        (padding_before, padding_after), 
                                                        'constant', 0)
            
            # 如果需要任何預處理，創建臨時檔案
            if needs_format_fix or needs_padding:
                input_path_obj = Path(input_path)
                temp_filename = f"processed_{int(time.time())}_{input_path_obj.name}"
                temp_path = self.temp_dir / temp_filename
                
                # 保存預處理後的音頻
                torchaudio.save(str(temp_path), waveform, sample_rate)
                
                if needs_padding:
                    self.logger.info(f"✅ 音頻補零完成: {duration:.2f}s → {self.target_duration:.2f}s")
                else:
                    self.logger.info(f"✅ 音頻格式標準化完成")
                
                return str(temp_path)
            else:
                return None  # 不需要預處理
            
        except Exception as e:
            self.logger.error(f"❌ 音頻預處理失敗 {input_path}: {e}")
            return None
    
    def enhance_audio(self, input_path: str, output_path: Optional[str] = None, 
                     backup_original: any = False) -> Dict:
        """
        對單個音檔進行 UVR5 增強處理
        
        Args:
            input_path: 輸入音檔路徑
            output_path: 輸出路徑 (None = 原地替換)
            backup_original: 是否備份原始檔案 (可接受 'true'/'false' 字串)
            
        Returns:
            Dict: 處理結果
        """
        # 將傳入的 backup_original (可能為字串 'true'/'false') 轉換為布林值
        backup_original = str(backup_original).lower() == 'true'

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
        
        original_duration = self.get_audio_duration(str(input_path))
        result['original_duration'] = original_duration
        
        preprocessed_file = None
        actual_input_path = input_path
        temp_output_dir = None
        
        try:
            # --- 主要處理邏輯 ---
            # 所有音檔都需要格式檢查和標準化（不只是短音檔）
            preprocessed_file = self.pad_audio_for_uvr5(str(input_path))
            if preprocessed_file:
                actual_input_path = Path(preprocessed_file)
                result['preprocessed'] = True
            else:
                self.logger.debug(f"ℹ️  音頻無需預處理: {input_path.name}")

            if backup_original and output_path == input_path:
                backup_path = input_path.with_suffix('.bak')
                
                # 檢查是否已存在備份檔案
                if backup_path.exists():
                    self.logger.info(f"ℹ️  備份檔案已存在，跳過備份: {backup_path.name}")
                    result['backup_skipped'] = True
                    result['backup_file'] = str(backup_path)
                    # 備份檔案已存在，使用備份檔案進行處理
                    if not preprocessed_file:
                        actual_input_path = backup_path
                else:
                    input_path.rename(backup_path)
                    input_path = backup_path
                    result['backup_file'] = str(backup_path)
                    self.logger.info(f"💾 已備份原始檔案: {backup_path.name}")
                    
                    # 更新actual_input_path，如果之前是預處理檔案則不變，否則更新為備份檔案
                    if not preprocessed_file:
                        actual_input_path = backup_path

            # 使用專用暫存目錄，避免在根目錄產生散落檔案
            temp_output_dir = self.temp_dir / f"uvr5_processing_{int(time.time())}"
            temp_output_dir.mkdir(exist_ok=True)
            
            self.separator.output_dir = str(temp_output_dir)
            output_files = self.separator.separate(str(actual_input_path))
            
            # 除錯：記錄返回的檔案列表
            self.logger.info(f"🔍 分離器回傳檔案列表: {output_files}")
            for f in output_files:
                # 檢查兩個可能的位置
                file_path1 = temp_output_dir / f
                file_path2 = self.temp_dir / f
                self.logger.info(f"  📁 檔案: {f}")
                self.logger.info(f"    位置1 {temp_output_dir}: {file_path1.exists()}")
                self.logger.info(f"    位置2 {self.temp_dir}: {file_path2.exists()}")
            
            vocals_file = next((f for f in output_files if 'vocals' in Path(f).name.lower() or '(vocals)' in Path(f).name.lower()), None)
            self.logger.info(f"🎤 找到的 Vocals 檔案: {vocals_file}")
            
            if vocals_file:
                # 檢查兩個可能的位置
                vocals_path1 = temp_output_dir / vocals_file
                vocals_path2 = self.temp_dir / vocals_file
                
                if vocals_path1.exists():
                    vocals_path = vocals_path1
                    self.logger.info(f"🎤 使用位置1: {vocals_path}")
                elif vocals_path2.exists():
                    vocals_path = vocals_path2
                    self.logger.info(f"🎤 使用位置2: {vocals_path}")
                else:
                    vocals_path = None
                    self.logger.error(f"❌ Vocals 檔案在兩個位置都不存在")
                
                if vocals_path:
                    # 如果進行了預處理（補零），需要還原到原始長度
                    if preprocessed_file and original_duration > 0:
                        self.logger.info(f"🔧 還原音頻長度: 15.00s → {original_duration:.2f}s")
                        
                        # 載入處理後的人聲檔案
                        processed_waveform, processed_sample_rate = torchaudio.load(str(vocals_path))
                        
                        # 計算原始音頻的樣本數
                        original_samples = int(original_duration * processed_sample_rate)
                        
                        # 從補零音頻中提取原始長度部分（移除前後補零）
                        # 補零是前後平均分配的，所以從中間提取原始長度
                        total_samples = processed_waveform.shape[1]
                        padding_samples = total_samples - original_samples
                        padding_before = padding_samples // 2
                        
                        # 提取原始音頻部分
                        restored_waveform = processed_waveform[:, padding_before:padding_before + original_samples]
                        
                        # 保存還原長度的音頻
                        temp_restored_file = temp_output_dir / f"restored_{Path(vocals_file).name}"
                        torchaudio.save(str(temp_restored_file), restored_waveform, processed_sample_rate)
                        
                        # 移動還原後的檔案到最終位置
                        shutil.move(str(temp_restored_file), str(output_path))
                        self.logger.info(f"✅ 人聲分離完成: {input_path.name} (還原: {original_duration:.2f}s)")
                    else:
                        # 沒有預處理的情況，直接移動
                        shutil.move(str(vocals_path), str(output_path))
                        self.logger.info(f"✅ 人聲分離完成: {input_path.name} (原始: {original_duration:.2f}s)")
                    
                    result['enhanced'] = True
            else:
                raise RuntimeError("人聲檔案生成失敗")

            result['success'] = True

        except Exception as e:
            # --- 錯誤處理 ---
            result['error'] = str(e)
            self.logger.error(f"❌ 人聲分離失敗 {input_path.name}: {e}")
            self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
            
            if backup_original and 'backup_file' in result:
                backup_path = Path(result['backup_file'])
                if backup_path.exists():
                    backup_path.rename(output_path)
        
        finally:
            # --- 資源清理 ---
            if temp_output_dir and temp_output_dir.exists():
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            
            if preprocessed_file and Path(preprocessed_file).exists():
                try:
                    Path(preprocessed_file).unlink()
                except OSError as e:
                    self.logger.warning(f"⚠️  清理預處理檔案失敗: {e}")
            
            # 額外清理：清除可能散落在工作目錄的伴奏檔案
            try:
                current_dir = Path.cwd()
                for pattern in ['*_(Instrumental)_*.wav', 'padded_*_(Instrumental)_*.wav']:
                    for leftover_file in current_dir.glob(pattern):
                        leftover_file.unlink()
                        self.logger.debug(f"🧹 清理散落檔案: {leftover_file.name}")
            except Exception as e:
                self.logger.debug(f"清理散落檔案時出現錯誤: {e}")
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # 更新最終統計數據
        result['processing_time'] = time.time() - start_time
        current_memory = psutil.Process().memory_info().rss / (1024**2)
        result['memory_usage_mb'] = current_memory - initial_memory
        
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
        
        # 尋找音檔 - 支援巢狀目錄搜尋 (rglob 會遞迴搜尋所有子目錄)
        audio_files = list(input_dir.rglob(pattern))
        if not audio_files:
            self.logger.warning(f"在 {input_dir} 中未找到匹配 {pattern} 的音檔")
            return {'success': False, 'error': 'No audio files found'}
        
        self.logger.info(f"📁 找到 {len(audio_files)} 個音檔進行處理")
        
        # 重置統計
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': [],
            'preprocessed_files': 0,
            'short_audio_count': 0,
            'average_original_duration': 0
        }
        
        start_time = time.time()
        
        # 批量處理
        for audio_file in tqdm(audio_files, desc="🎵 UVR5 人聲分離"):
            try:
                result = self.enhance_audio(str(audio_file), backup_original=backup_original)
                
                if result['success']:
                    self.stats['processed_files'] += 1
                    # 統計預處理資訊
                    if result.get('preprocessed', False):
                        self.stats['preprocessed_files'] += 1
                    if result.get('original_duration', 0) > 0 and result.get('original_duration', 0) < self.min_duration:
                        self.stats['short_audio_count'] += 1
                    
                    # 記錄成功處理的詳細信息
                    processing_time = result.get('processing_time', 0)
                    memory_usage = result.get('memory_usage_mb', 0)
                    self.logger.debug(f"✅ 成功處理 {audio_file.name}: {processing_time:.2f}s, {memory_usage:.1f}MB")
                else:
                    self.stats['failed_files'] += 1
                    error_msg = result.get('error', 'Unknown error')
                    self.stats['failed_list'].append({
                        'file': str(audio_file),
                        'error': error_msg,
                        'processing_time': result.get('processing_time', 0)
                    })
                    self.logger.error(f"❌ 處理失敗 {audio_file.name}: {error_msg}")
                
            except Exception as e:
                self.stats['failed_files'] += 1
                error_msg = f"Unexpected error: {str(e)}"
                self.stats['failed_list'].append({
                    'file': str(audio_file),
                    'error': error_msg,
                    'exception_type': type(e).__name__
                })
                self.logger.error(f"❌ 處理失敗 {audio_file.name}: {error_msg}")
                # 記錄完整的錯誤堆疊追蹤
                self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
        
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
                    # 記錄完整的錯誤堆疊追蹤
                    self.logger.debug(f"Exception traceback: {traceback.format_exc()}")
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
            'batch_size': self.batch_size,
            'min_duration': self.min_duration,
            'target_duration': self.target_duration,
            'processing_timeout': self.processing_timeout
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
        # 記錄最終的 GPU 記憶體狀態
        if self.device == 'cuda' and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            self.logger.debug(f"📊 GPU 記憶體清理後: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")


class ThreadedUVR5Processor(UVR5Processor):
    """多執行緒 UVR5 處理器 - 支援並行處理以提升大批量檔案的處理速度"""
    
    def __init__(self, max_workers: int = 1, **kwargs):
        """
        初始化多執行緒 UVR5 處理器
        
        Args:
            max_workers: 最大並行執行緒數，1=單執行緒，2+=多執行緒
            **kwargs: 傳遞給父類別的其他參數
        """
        super().__init__(**kwargs)
        self.max_workers = max(1, int(max_workers))  # 確保至少為 1
        self.logger.info(f"🚀 多執行緒 UVR5 處理器初始化完成，並行數: {self.max_workers}")
    
    def batch_enhance(self, input_dir: str, pattern: str = "*.wav",
                     backup_original: bool = False) -> Dict:
        """
        多執行緒批量處理目錄下的音檔
        
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
        
        # 尋找音檔 - 支援巢狀目錄搜尋 (rglob 會遞迴搜尋所有子目錄)
        audio_files = list(input_dir.rglob(pattern))
        if not audio_files:
            self.logger.warning(f"在 {input_dir} 中未找到匹配 {pattern} 的音檔")
            return {'success': False, 'error': 'No audio files found'}
        
        total_files = len(audio_files)
        self.logger.info(f"📁 找到 {total_files} 個音檔進行處理")
        
        # 根據並行數選擇處理方式
        if self.max_workers <= 1:
            self.logger.info("🔄 使用單執行緒模式處理")
            return self._single_thread_batch_enhance(audio_files, backup_original)
        else:
            self.logger.info(f"🚀 使用多執行緒模式處理，並行數: {self.max_workers}")
            return self._multi_thread_batch_enhance(audio_files, backup_original)
    
    def _single_thread_batch_enhance(self, audio_files: List[Path], backup_original: bool) -> Dict:
        """單執行緒批量處理（原有邏輯）"""
        # 重置統計
        self.stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': [],
            'preprocessed_files': 0,
            'short_audio_count': 0
        }
        
        start_time = time.time()
        
        # 批量處理
        for audio_file in tqdm(audio_files, desc="🎵 UVR5 人聲分離"):
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
    
    def _multi_thread_batch_enhance(self, audio_files: List[Path], backup_original: bool) -> Dict:
        """多執行緒批量處理"""
        start_time = time.time()
        
        # 檢查 GPU 記憶體
        if not self._check_gpu_memory():
            self.logger.warning("⚠️  GPU 記憶體不足，降級到單執行緒模式")
            return self._single_thread_batch_enhance(audio_files, backup_original)
        
        # 初始化統計
        stats = {
            'processed_files': 0,
            'failed_files': 0,
            'total_time': 0,
            'failed_list': []
        }
        
        # 創建執行緒池和 UVR5 處理器實例
        processors = []
        try:
            self.logger.info(f"🔧 創建 {self.max_workers} 個 UVR5 處理器實例...")
            for i in range(self.max_workers):
                processor = UVR5Processor(
                    model_path=str(self.model_path),
                    vocal_model=self.vocal_model,
                    device=self.device,
                    batch_size=self.batch_size,
                    min_duration=self.min_duration,
                    target_duration=self.target_duration
                )
                processors.append(processor)
            
            # 多執行緒處理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任務
                future_to_file = {}
                for i, audio_file in enumerate(audio_files):
                    processor = processors[i % self.max_workers]
                    future = executor.submit(
                        processor.enhance_audio, 
                        str(audio_file), 
                        backup_original=backup_original
                    )
                    future_to_file[future] = audio_file
                
                # 收集結果並顯示進度
                progress_bar = tqdm(total=len(audio_files), desc="🚀 多執行緒 UVR5 人聲分離")
                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result['success']:
                            stats['processed_files'] += 1
                        else:
                            stats['failed_files'] += 1
                            stats['failed_list'].append({
                                'file': str(audio_file),
                                'error': result.get('error', 'Unknown error')
                            })
                    except Exception as e:
                        stats['failed_files'] += 1
                        stats['failed_list'].append({
                            'file': str(audio_file),
                            'error': str(e)
                        })
                        self.logger.error(f"❌ 多執行緒處理失敗 {audio_file.name}: {e}")
                    
                    progress_bar.update(1)
                
                progress_bar.close()
            
        finally:
            # 清理所有處理器實例
            for processor in processors:
                try:
                    processor.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️  清理處理器時出錯: {e}")
                    self.logger.debug(f"Cleanup exception traceback: {traceback.format_exc()}")
        
        stats['total_time'] = time.time() - start_time
        
        # 生成報告
        self._generate_threaded_batch_report(stats, len(audio_files))
        
        return {
            'success': True,
            'stats': stats,
            'total_files': len(audio_files),
            'processed_files': stats['processed_files'],
            'failed_files': stats['failed_files']
        }
    
    def _check_gpu_memory(self) -> bool:
        """檢查 GPU 記憶體是否足夠進行多執行緒處理"""
        if self.device != 'cuda' or not torch.cuda.is_available():
            return True  # CPU 模式不需要檢查
        
        try:
            # 獲取 GPU 記憶體資訊
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_free = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            
            # 估算每個處理器需要的記憶體 (保守估計 2.5GB)
            estimated_memory_per_worker = 2.5 * 1024**3  # 2.5GB
            required_memory = estimated_memory_per_worker * self.max_workers
            
            self.logger.info(f"📊 GPU 記憶體檢查:")
            self.logger.info(f"  總記憶體: {gpu_memory / 1024**3:.1f} GB")
            self.logger.info(f"  可用記憶體: {gpu_free / 1024**3:.1f} GB")
            self.logger.info(f"  需要記憶體: {required_memory / 1024**3:.1f} GB ({self.max_workers} 執行緒)")
            
            if gpu_free < required_memory:
                self.logger.warning(f"⚠️  GPU 記憶體可能不足，建議降低並行數或使用單執行緒模式")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  無法檢查 GPU 記憶體: {e}")
            return True  # 檢查失敗時允許繼續
            
        except Exception as e:
            self.logger.error(f"❌ GPU 記憶體檢查出現意外錯誤: {e}")
            self.logger.debug(f"GPU memory check exception: {traceback.format_exc()}")
            return True  # 出現意外錯誤時允許繼續
    
    def _generate_threaded_batch_report(self, stats: Dict, total_files: int):
        """生成多執行緒批量處理報告"""
        print("\n" + "="*60)
        print("🚀 多執行緒 UVR5 人聲分離批量處理報告")
        print("="*60)
        print(f"📊 處理統計:")
        print(f"  並行執行緒數: {self.max_workers}")
        print(f"  成功處理: {stats['processed_files']} 檔案")
        print(f"  處理失敗: {stats['failed_files']} 檔案")
        print(f"  總處理時間: {stats['total_time']:.2f} 秒")
        
        if stats['processed_files'] > 0:
            avg_time = stats['total_time'] / stats['processed_files']
            print(f"  平均處理時間: {avg_time:.2f} 秒/檔")
            
            # 估算加速比（相對於單執行緒）
            estimated_single_thread_time = stats['total_time'] * self.max_workers
            speedup = estimated_single_thread_time / stats['total_time']
            print(f"  估算加速比: {speedup:.1f}x")
        
        if stats['failed_files'] > 0:
            print(f"\n❌ 失敗檔案清單:")
            for failed in stats['failed_list']:
                print(f"  • {failed['file']}: {failed['error']}")


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