#!/usr/bin/env python3
"""
整合 Pipeline：UVR5 去背 + Pyannote 語者分離 + 分段處理
一條龍音頻處理服務
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import traceback

# 本地模組
from .uvr5_vocal_separator import UVR5VocalSeparator
from .offline_pipeline import OfflinePipelineLoader
from .pyannote_speaker_segmentation import load_pipeline_and_process

# 靜音警告
warnings.filterwarnings("ignore")


class IntegratedAudioPipeline:
    """整合音頻處理 Pipeline"""
    
    def __init__(self, 
                 project_root: Optional[str] = None,
                 enable_uvr5: bool = True,
                 enable_speaker_diarization: bool = True,
                 use_gpu: bool = True):
        """
        初始化整合 Pipeline
        
        Args:
            project_root: 專案根目錄
            enable_uvr5: 是否啟用 UVR5 去背
            enable_speaker_diarization: 是否啟用語者分離
            use_gpu: 是否使用 GPU
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.enable_uvr5 = enable_uvr5
        self.enable_speaker_diarization = enable_speaker_diarization
        self.use_gpu = use_gpu
        
        # 設定目錄
        self.setup_directories()
        
        # 設定 logging
        self.setup_logging()
        
        # 初始化組件
        self.uvr5_separator = None
        self.pyannote_pipeline = None
        self.pyannote_device = None
        
        # 載入組件
        self.initialize_components()
    
    def setup_directories(self):
        """設定必要目錄"""
        self.models_dir = self.project_root / "models"
        self.uvr5_models_dir = self.models_dir / "uvr5"
        self.output_dir = self.project_root / "data" / "processed"
        self.temp_dir = self.project_root / "data" / "temp"
        
        # 確保目錄存在
        for directory in [self.uvr5_models_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """設定 logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_root / 'integrated_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_components(self):
        """初始化所有組件"""
        try:
            # 初始化 UVR5 去背模組
            if self.enable_uvr5:
                self.logger.info("🎵 初始化 UVR5 去背模組...")
                self.uvr5_separator = UVR5VocalSeparator(
                    models_dir=str(self.uvr5_models_dir),
                    output_dir=str(self.temp_dir / "separated"),
                    use_gpu=self.use_gpu
                )
            
            # 初始化 Pyannote 語者分離模組
            if self.enable_speaker_diarization:
                self.logger.info("👥 初始化 Pyannote 語者分離模組...")
                loader = OfflinePipelineLoader(self.project_root)
                self.pyannote_pipeline = loader.load_pipeline()
                self.pyannote_device = loader.setup_gpu_if_available(self.pyannote_pipeline)
            
            self.logger.info("✅ 所有組件初始化完成")
            
        except Exception as e:
            self.logger.error(f"❌ 組件初始化失敗: {e}")
            raise
    
    def process_audio_file(self, 
                          input_file: str,
                          output_prefix: Optional[str] = None,
                          uvr5_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                          min_speakers: int = 1,
                          max_speakers: int = 10) -> Dict[str, Any]:
        """
        處理單個音頻文件的完整流程
        
        Args:
            input_file: 輸入音頻文件路徑
            output_prefix: 輸出文件前綴
            uvr5_model: UVR5 模型名稱
            min_speakers: 最小語者數量
            max_speakers: 最大語者數量
            
        Returns:
            Dict: 處理結果
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"輸入文件不存在: {input_path}")
        
        if output_prefix is None:
            output_prefix = input_path.stem
        
        start_time = time.time()
        
        result = {
            'input_file': str(input_path),
            'output_prefix': output_prefix,
            'processing_stages': {},
            'final_outputs': {},
            'total_time': 0,
            'success': False,
            'error': None
        }
        
        try:
            self.logger.info(f"🚀 開始處理音頻文件: {input_path.name}")
            
            # 步驟 1: UVR5 去背（如果啟用）
            vocals_file = str(input_path)
            if self.enable_uvr5:
                vocals_file = self._step_1_uvr5_separation(
                    str(input_path), output_prefix, uvr5_model, result
                )
            
            # 步驟 2: Pyannote 語者分離（如果啟用）
            if self.enable_speaker_diarization and vocals_file:
                self._step_2_speaker_diarization(
                    vocals_file, output_prefix, min_speakers, max_speakers, result
                )
            
            # 步驟 3: 整理最終輸出
            self._step_3_organize_outputs(output_prefix, result)
            
            result['total_time'] = time.time() - start_time
            result['success'] = True
            
            self.logger.info(f"✅ 音頻處理完成: {input_path.name} ({result['total_time']:.2f}秒)")
            
        except Exception as e:
            result['error'] = str(e)
            result['total_time'] = time.time() - start_time
            self.logger.error(f"❌ 音頻處理失敗 {input_path}: {e}")
            if self.logger.level <= logging.DEBUG:
                self.logger.debug(traceback.format_exc())
        
        return result
    
    def _step_1_uvr5_separation(self, input_file: str, output_prefix: str, 
                                model_name: str, result: Dict) -> Optional[str]:
        """步驟 1: UVR5 音頻去背"""
        self.logger.info("🎵 步驟 1: 執行 UVR5 音頻去背...")
        
        if self.uvr5_separator is None:
            raise RuntimeError("UVR5 分離器未初始化")
        
        # 初始化分離器（如果還沒有）
        if self.uvr5_separator.separator is None:
            self.uvr5_separator.initialize_separator(model_name)
        
        # 執行去背
        uvr5_result = self.uvr5_separator.separate_vocals(input_file, output_prefix)
        result['processing_stages']['uvr5'] = uvr5_result
        
        if uvr5_result['success']:
            vocals_file = uvr5_result['output_files'].get('vocals')
            self.logger.info(f"✅ UVR5 去背完成，人聲文件: {vocals_file}")
            return vocals_file
        else:
            self.logger.error(f"❌ UVR5 去背失敗: {uvr5_result.get('error', 'Unknown error')}")
            return None
    
    def _step_2_speaker_diarization(self, vocals_file: str, output_prefix: str,
                                   min_speakers: int, max_speakers: int, result: Dict):
        """步驟 2: Pyannote 語者分離"""
        self.logger.info("👥 步驟 2: 執行 Pyannote 語者分離...")
        
        if self.pyannote_pipeline is None:
            raise RuntimeError("Pyannote Pipeline 未初始化")
        
        # 設定輸出目錄
        speaker_output_dir = self.output_dir / output_prefix
        speaker_output_dir.mkdir(exist_ok=True)
        
        try:
            # 執行語者分離
            diarization_result = self.pyannote_pipeline(vocals_file)
            
            # 處理語者分離結果
            segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
            
            # 儲存語者分離結果
            import json
            results_file = speaker_output_dir / f"{output_prefix}_diarization.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'input_file': vocals_file,
                    'total_segments': len(segments),
                    'unique_speakers': len(set(seg['speaker'] for seg in segments)),
                    'segments': segments
                }, f, indent=2, ensure_ascii=False)
            
            result['processing_stages']['diarization'] = {
                'success': True,
                'input_file': vocals_file,
                'output_file': str(results_file),
                'total_segments': len(segments),
                'unique_speakers': len(set(seg['speaker'] for seg in segments)),
                'segments': segments
            }
            
            self.logger.info(f"✅ 語者分離完成，找到 {len(set(seg['speaker'] for seg in segments))} 個語者")
            
        except Exception as e:
            result['processing_stages']['diarization'] = {
                'success': False,
                'error': str(e)
            }
            self.logger.error(f"❌ 語者分離失敗: {e}")
            raise
    
    def _step_3_organize_outputs(self, output_prefix: str, result: Dict):
        """步驟 3: 整理最終輸出"""
        self.logger.info("📁 步驟 3: 整理最終輸出...")
        
        final_output_dir = self.output_dir / output_prefix
        final_output_dir.mkdir(exist_ok=True)
        
        final_outputs = {
            'output_directory': str(final_output_dir)
        }
        
        # UVR5 輸出
        if 'uvr5' in result['processing_stages'] and result['processing_stages']['uvr5']['success']:
            uvr5_outputs = result['processing_stages']['uvr5']['output_files']
            final_outputs['separated_vocals'] = uvr5_outputs.get('vocals')
            final_outputs['separated_instrumental'] = uvr5_outputs.get('instrumental')
        
        # 語者分離輸出
        if 'diarization' in result['processing_stages'] and result['processing_stages']['diarization']['success']:
            final_outputs['diarization_results'] = result['processing_stages']['diarization']['output_file']
            final_outputs['speaker_count'] = result['processing_stages']['diarization']['unique_speakers']
        
        result['final_outputs'] = final_outputs
        
        self.logger.info(f"📋 最終輸出整理完成，文件位於: {final_output_dir}")
    
    def batch_process(self, 
                     input_files: List[str],
                     output_prefix: str = "batch",
                     **kwargs) -> List[Dict]:
        """
        批次處理多個音頻文件
        
        Args:
            input_files: 輸入文件列表
            output_prefix: 輸出前綴
            **kwargs: 其他處理參數
            
        Returns:
            List[Dict]: 處理結果列表
        """
        results = []
        
        self.logger.info(f"🚀 開始批次處理 {len(input_files)} 個文件")
        
        for i, input_file in enumerate(input_files, 1):
            self.logger.info(f"📁 處理第 {i}/{len(input_files)} 個文件: {Path(input_file).name}")
            
            file_prefix = f"{output_prefix}_{i:03d}_{Path(input_file).stem}"
            result = self.process_audio_file(input_file, file_prefix, **kwargs)
            results.append(result)
        
        # 生成批次處理報告
        self._generate_batch_report(results)
        
        return results
    
    def _generate_batch_report(self, results: List[Dict]):
        """生成批次處理報告"""
        total_files = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_files - successful
        
        total_time = sum(r.get('total_time', 0) for r in results)
        avg_time = total_time / total_files if total_files > 0 else 0
        
        print("\n" + "="*60)
        print("整合音頻處理 Pipeline 批次處理報告")
        print("="*60)
        print(f"總檔案數: {total_files}")
        print(f"成功處理: {successful}")
        print(f"失敗檔案: {failed}")
        print(f"成功率: {(successful/total_files*100):.1f}%")
        print(f"總處理時間: {total_time:.2f} 秒")
        print(f"平均處理時間: {avg_time:.2f} 秒/檔")
        
        # 統計各階段成功率
        if self.enable_uvr5:
            uvr5_success = sum(1 for r in results if r.get('processing_stages', {}).get('uvr5', {}).get('success', False))
            print(f"UVR5 去背成功率: {(uvr5_success/total_files*100):.1f}%")
        
        if self.enable_speaker_diarization:
            diarization_success = sum(1 for r in results if r.get('processing_stages', {}).get('diarization', {}).get('success', False))
            print(f"語者分離成功率: {(diarization_success/total_files*100):.1f}%")
    
    def cleanup(self):
        """清理資源"""
        if self.uvr5_separator:
            self.uvr5_separator.cleanup()
        
        # 清理暫存文件
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(exist_ok=True)
        
        self.logger.info("🧹 Pipeline 資源清理完成")


def create_integrated_pipeline(**kwargs) -> IntegratedAudioPipeline:
    """
    便利函數：創建整合音頻處理 Pipeline
    
    Args:
        **kwargs: Pipeline 初始化參數
        
    Returns:
        IntegratedAudioPipeline: 整合 Pipeline 實例
    """
    return IntegratedAudioPipeline(**kwargs)


def test_integrated_pipeline():
    """測試整合 Pipeline"""
    print("🎯 測試整合音頻處理 Pipeline")
    print("=" * 50)
    
    try:
        # 創建 Pipeline
        pipeline = create_integrated_pipeline()
        
        print("✅ 整合 Pipeline 測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False
    
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()


if __name__ == "__main__":
    success = test_integrated_pipeline()
    exit(0 if success else 1)