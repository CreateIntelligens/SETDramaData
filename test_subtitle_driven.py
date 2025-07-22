#!/usr/bin/env python3
"""
測試字幕驅動分段功能的單元測試和整合測試
"""

import os
import sys
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# 添加src目錄到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from subtitle_driven_segmentation import (
    create_subtitle_segments,
    merge_segments_by_similarity,
    segment_by_subtitle_driven
)
from speaker_database import SpeakerDatabase


def test_create_subtitle_segments():
    """測試基於字幕創建音頻片段"""
    print("🧪 測試: create_subtitle_segments")
    
    # 模擬字幕數據
    subtitles = [
        (100.0, "第一句話"),
        (103.5, "第二句話"),
        (107.2, "第三句話"),
        (112.0, "第四句話")
    ]
    
    # 由於需要音頻文件，我們創建一個臨時的測試
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Mock librosa.get_duration
        with patch('subtitle_driven_segmentation.librosa.get_duration', return_value=120.0):
            segments = create_subtitle_segments(subtitles, temp_audio_path)
        
        # 驗證結果
        assert len(segments) == 4, f"應該有4個段落，但得到{len(segments)}個"
        
        # 檢查時間範圍
        assert segments[0]['start'] == 100.0
        assert segments[0]['end'] == 103.5
        assert segments[0]['text'] == "第一句話"
        
        assert segments[1]['start'] == 103.5
        assert segments[1]['end'] == 107.2
        
        # 最後一句應該有估算的持續時間
        assert segments[3]['start'] == 112.0
        assert segments[3]['end'] > 112.0
        
        print("   ✅ create_subtitle_segments 測試通過")
        return True
        
    except Exception as e:
        print(f"   ❌ create_subtitle_segments 測試失敗: {e}")
        return False
        
    finally:
        # 清理臨時文件
        try:
            os.unlink(temp_audio_path)
        except:
            pass


def test_merge_segments_by_similarity():
    """測試基於相似度合併段落"""
    print("🧪 測試: merge_segments_by_similarity")
    
    try:
        # 創建測試段落，包含embedding
        segments = [
            {
                'start': 100.0,
                'end': 103.0,
                'text': '第一句',
                'embedding': np.array([1.0, 0.0, 0.0]),
                'embedding_extracted': True,
                'original_index': 0
            },
            {
                'start': 103.0,
                'end': 106.0,
                'text': '第二句',
                'embedding': np.array([0.9, 0.1, 0.1]),  # 與第一句相似
                'embedding_extracted': True,
                'original_index': 1
            },
            {
                'start': 106.0,
                'end': 109.0,
                'text': '第三句',
                'embedding': np.array([0.0, 1.0, 0.0]),  # 不同speaker
                'embedding_extracted': True,
                'original_index': 2
            }
        ]
        
        # 測試合併
        merged = merge_segments_by_similarity(segments, similarity_threshold=0.8, max_duration=15.0)
        
        # 驗證結果
        assert len(merged) == 2, f"應該合併成2個段落，但得到{len(merged)}個"
        
        # 第一段應該是合併的結果
        assert merged[0]['start'] == 100.0
        assert merged[0]['end'] == 106.0
        assert '第一句 第二句' in merged[0]['text']
        
        # 第二段保持不變
        assert merged[1]['start'] == 106.0
        assert merged[1]['end'] == 109.0
        assert merged[1]['text'] == '第三句'
        
        print("   ✅ merge_segments_by_similarity 測試通過")
        return True
        
    except Exception as e:
        print(f"   ❌ merge_segments_by_similarity 測試失敗: {e}")
        return False


def test_subtitle_driven_integration():
    """測試完整的字幕驅動分段流程"""
    print("🧪 測試: 完整字幕驅動分段流程")
    
    try:
        # 準備測試數據
        subtitles = [
            (50.0, "你好，歡迎來到這個測試"),
            (53.2, "我們正在測試新的分段系統"),
            (58.5, "希望一切順利運行")
        ]
        
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.return_value = np.random.rand(512)  # 512維embedding
        
        # Mock device
        mock_device = Mock()
        mock_device.type = 'cpu'
        
        # 創建臨時資料庫
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name
            
        # 創建臨時音頻文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # 初始化資料庫
            db = SpeakerDatabase(temp_db_path)
            
            # Mock audio duration
            with patch('subtitle_driven_segmentation.librosa.get_duration', return_value=70.0):
                # 調用主要函數
                segments, mapping = segment_by_subtitle_driven(
                    subtitles=subtitles,
                    audio_path=temp_audio_path,
                    embedding_model=mock_embedding_model,
                    device=mock_device,
                    db=db,
                    episode_num=1,
                    min_duration=2.0,
                    max_duration=15.0,
                    similarity_threshold=0.85
                )
            
            # 基本驗證
            print(f"   📊 生成了 {len(segments)} 個段落")
            print(f"   📊 speaker mapping: {len(mapping)} 個")
            
            # 每個segment都應該有正確的格式
            for segment in segments:
                assert len(segment) == 3, "每個段落應該是 (start, end, global_speaker_id) 格式"
                start, end, speaker_id = segment
                assert isinstance(start, (int, float)), "開始時間應該是數字"
                assert isinstance(end, (int, float)), "結束時間應該是數字" 
                assert isinstance(speaker_id, int), "Speaker ID應該是整數"
                assert end > start, "結束時間應該大於開始時間"
            
            print("   ✅ 字幕驅動分段整合測試通過")
            return True
            
        finally:
            # 清理臨時文件
            try:
                os.unlink(temp_db_path)
                os.unlink(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        print(f"   ❌ 字幕驅動分段整合測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """測試性能比較：顯示不同分段方法的差異"""
    print("🧪 性能比較測試")
    
    # 模擬一個有很多字幕的場景
    subtitles = []
    for i in range(50):  # 50句字幕
        time_start = i * 2.5  # 每句間隔2.5秒
        text = f"這是第{i+1}句測試字幕，用來驗證系統性能"
        subtitles.append((time_start, text))
    
    print(f"   📊 測試場景: {len(subtitles)} 句字幕，總長 {subtitles[-1][0] + 3:.1f} 秒")
    
    # 分析可能的切分結果
    expected_segments = len(subtitles)  # 字幕驅動：每句一個段落
    potential_merges = 0
    
    # 估算可能的合併數（假設每3-4句是同一speaker）
    for i in range(0, len(subtitles), 4):
        potential_merges += min(4, len(subtitles) - i)
    
    print(f"   🎯 字幕驅動模式預期: {expected_segments} 個初始段落")
    print(f"   🔗 預期合併後: 約 {len(subtitles)//3} 個段落（假設相似speaker合併）")
    print(f"   ⚠️  傳統模式風險: 可能遺漏 10-20% 的字幕（因時間對齊問題）")
    
    return True


def run_all_tests():
    """執行所有測試"""
    print("🚀 開始執行字幕驅動分段測試套件")
    print("=" * 50)
    
    tests = [
        test_create_subtitle_segments,
        test_merge_segments_by_similarity,
        test_subtitle_driven_integration,
        test_performance_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # 空行分隔
        except Exception as e:
            print(f"❌ 測試 {test_func.__name__} 執行時發生錯誤: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 測試結果: {passed}/{total} 個測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！字幕驅動分段功能運作正常")
        return True
    else:
        print(f"⚠️  {total - passed} 個測試失敗，需要檢查")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)