#!/usr/bin/env python3
"""
字幕驅動的speaker分段邏輯
- 基於字幕時間軸預切分音頻
- 使用embedding比較相鄰片段決定是否合併
- 確保不會遺漏字幕內容
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import torch
from tqdm import tqdm


def segment_by_subtitle_driven(
    subtitles: List[Tuple[float, str]],
    audio_path: str,
    embedding_model,
    device: torch.device,
    db,  # 資料庫物件
    episode_num: int,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
    similarity_threshold: float = 0.85
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """
    基於字幕時間軸的speaker分段，然後用embedding合併
    
    Args:
        subtitles: 字幕列表 [(timestamp, text), ...]
        audio_path: 音頻檔案路徑
        embedding_model: speaker embedding模型
        device: 計算設備
        db: speaker資料庫
        episode_num: 集數
        min_duration: 最小段落長度
        max_duration: 最大段落長度
        similarity_threshold: speaker相似度閾值
        
    Returns:
        (segments_with_global_ids, local_to_global_mapping)
    """
    
    if not subtitles:
        print("   ❌ 沒有字幕資料")
        return [], {}
    
    print(f"   📚 字幕驅動分段模式：{len(subtitles)} 句字幕")
    
    # Step 1: 建立初始的字幕音頻片段
    subtitle_segments = create_subtitle_segments(subtitles, audio_path)
    
    # Add audio_path to each segment for embedding extraction
    for segment in subtitle_segments:
        segment['audio_path'] = audio_path
    print(f"   ✂️ 建立了 {len(subtitle_segments)} 個字幕音頻片段")
    
    # Step 2: 為每個片段提取embedding
    segments_with_embeddings = extract_embeddings_for_segments(
        subtitle_segments, embedding_model, device
    )
    print(f"   🧬 提取了 {len(segments_with_embeddings)} 個embedding")
    
    # Step 3: 基於embedding相似度合併相鄰片段
    merged_segments = merge_segments_by_similarity(
        segments_with_embeddings, similarity_threshold, max_duration
    )
    print(f"   🔗 合併後剩餘 {len(merged_segments)} 個段落")
    
    # Step 4: 過濾長度並分配global speaker ID
    final_segments, local_to_global_map = assign_global_speaker_ids(
        merged_segments, db, episode_num, min_duration, max_duration, similarity_threshold
    )
    
    print(f"   ✅ 最終產生 {len(final_segments)} 個有效段落")
    return final_segments, local_to_global_map


def create_subtitle_segments(
    subtitles: List[Tuple[float, str]], 
    audio_path: str
) -> List[Dict]:
    """
    根據字幕時間軸創建音頻片段
    
    每個字幕的持續時間 = 到下一個字幕開始的時間間隔
    """
    import librosa
    
    # 獲取音頻總長度
    audio_duration = librosa.get_duration(path=audio_path)
    
    segments = []
    for i, (start_time, text) in enumerate(subtitles):
        # 計算這句字幕的結束時間
        if i < len(subtitles) - 1:
            # 不是最後一句：結束時間 = 下一句開始時間
            end_time = subtitles[i + 1][0]
        else:
            # 最後一句：估算持續時間（每字0.3秒，最少2秒，最多8秒）
            estimated_duration = max(2.0, min(8.0, len(text) * 0.3))
            end_time = min(start_time + estimated_duration, audio_duration)
        
        # 確保時間範圍合理
        if end_time > start_time + 0.5:  # 至少0.5秒
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': text,
                'original_index': i,
                'duration': end_time - start_time
            })
            print(f"     📝 #{i+1}: [{start_time:.1f}-{end_time:.1f}s] ({end_time-start_time:.1f}s) '{text}'")
        else:
            print(f"     ⚠️ 跳過過短字幕 #{i+1}: {text}")
    
    return segments


def extract_embeddings_for_segments(
    segments: List[Dict],
    embedding_model,
    device: torch.device
) -> List[Dict]:
    """為每個字幕音頻片段提取speaker embedding"""
    
    segments_with_embeddings = []
    
    for i, segment in enumerate(tqdm(segments, desc="   提取embeddings", unit="seg", ncols=80)):
        start, end, text = segment['start'], segment['end'], segment['text']
        
        # 創建segment dict給pyannote
        segment_dict = {
            "audio": segment['audio_path'],
            "start": start,
            "end": end
        }
        
        try:
            with torch.no_grad():
                embedding_output = embedding_model(segment_dict)
            
            # 處理不同類型的embedding輸出
            if hasattr(embedding_output, 'cpu'):
                embedding_np = embedding_output.cpu().numpy()
            elif isinstance(embedding_output, dict):
                if 'embeddings' in embedding_output:
                    embedding_np = embedding_output['embeddings']
                elif 'embedding' in embedding_output:
                    embedding_np = embedding_output['embedding']
                else:
                    embedding_np = list(embedding_output.values())[0]
                
                # 轉換為numpy
                if hasattr(embedding_np, 'cpu'):
                    embedding_np = embedding_np.cpu().numpy()
            else:
                embedding_np = embedding_output
            
            # 確保是1D embedding
            if embedding_np.ndim > 1:
                embedding_np = embedding_np.flatten()
            
            # 複製segment資料並加入embedding
            segment_copy = segment.copy()
            segment_copy['embedding'] = embedding_np
            segment_copy['embedding_extracted'] = True
            segments_with_embeddings.append(segment_copy)
            
            print(f"     ✅ #{i+1}: embedding shape {embedding_np.shape}, mean={np.mean(embedding_np):.4f}")
            
        except Exception as e:
            print(f"     ❌ #{i+1}: embedding提取失敗: {e}")
            # 添加空embedding
            segment_copy = segment.copy()
            segment_copy['embedding'] = np.zeros(512)
            segment_copy['embedding_extracted'] = False
            segments_with_embeddings.append(segment_copy)
    
    return segments_with_embeddings


def merge_segments_by_similarity(
    segments: List[Dict],
    similarity_threshold: float,
    max_duration: float
) -> List[Dict]:
    """基於embedding相似度合併相鄰的音頻片段"""
    
    if len(segments) <= 1:
        return segments
    
    merged_segments = []
    current_segment = segments[0].copy()
    merge_count = 0
    
    print(f"   🔍 開始基於相似度合併 (閾值: {similarity_threshold})")
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # 檢查是否可以合併
        can_merge = False
        similarity = 0.0
        
        # 需要兩個segment都有有效的embedding
        if (current_segment.get('embedding_extracted', False) and 
            next_segment.get('embedding_extracted', False)):
            
            # 計算相似度
            emb1 = current_segment['embedding']
            emb2 = next_segment['embedding']
            
            try:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # 檢查合併條件
                merged_duration = next_segment['end'] - current_segment['start']
                
                can_merge = (
                    similarity > similarity_threshold and 
                    merged_duration <= max_duration
                )
                
                print(f"     🔍 比較段落 {current_segment['original_index']+1} vs {next_segment['original_index']+1}: "
                      f"相似度={similarity:.3f}, 合併長度={merged_duration:.1f}s "
                      f"→ {'✅合併' if can_merge else '❌分開'}")
                
            except Exception as e:
                print(f"     ⚠️ 相似度計算失敗: {e}")
                can_merge = False
        
        if can_merge:
            # 合併到current_segment
            current_segment['end'] = next_segment['end']
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            current_segment['text'] += " " + next_segment['text']
            
            # 使用平均embedding
            if ('embedding' in current_segment and 'embedding' in next_segment and
                current_segment.get('embedding_extracted', False) and 
                next_segment.get('embedding_extracted', False)):
                current_segment['embedding'] = (current_segment['embedding'] + next_segment['embedding']) / 2
            
            merge_count += 1
            print(f"     ✅ 合併成功: [{current_segment['start']:.1f}-{current_segment['end']:.1f}s] '{current_segment['text'][:50]}...'")
        else:
            # 無法合併，保存current並開始新的
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    # 添加最後一個段落
    merged_segments.append(current_segment)
    
    print(f"   📊 合併統計: 執行了 {merge_count} 次合併，最終 {len(merged_segments)} 個段落")
    return merged_segments


def assign_global_speaker_ids(
    segments: List[Dict],
    db,
    episode_num: int,
    min_duration: float,
    max_duration: float,
    similarity_threshold: float
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """為每個段落分配global speaker ID並過濾長度"""
    
    valid_segments = []
    local_to_global_map = {}
    
    print(f"   🎯 分配Global Speaker IDs (相似度閾值: {similarity_threshold})")
    
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        duration = end - start
        
        print(f"     處理段落 #{i+1}: [{start:.1f}-{end:.1f}s] ({duration:.1f}s)")
        
        # 過濾長度
        if not (min_duration <= duration <= max_duration):
            print(f"       ❌ 長度不合格 (需要 {min_duration}-{max_duration}s)")
            continue
        
        # 檢查是否有有效embedding
        if not segment.get('embedding_extracted', False):
            print(f"       ❌ 無有效embedding")
            continue
        
        embedding = segment['embedding']
        local_label = f"subtitle_seg_{i}"
        
        # 在資料庫中查找或註冊speaker
        try:
            speaker_id, similarity = db.find_similar_speaker(embedding, similarity_threshold)
            
            if speaker_id is not None:
                print(f"       🔍 匹配現有Global Speaker: {speaker_id} (相似度: {similarity:.3f})")
                db.update_speaker_episode(speaker_id, episode_num, local_label, 1)
            else:
                speaker_id = db.add_speaker(embedding, episode_num, local_label, 1)
                print(f"       ✨ 註冊新Global Speaker: {speaker_id}")
            
            # 記錄mapping並添加到結果
            local_to_global_map[local_label] = speaker_id
            valid_segments.append((start, end, speaker_id))
            
            print(f"       ✅ 段落已保存: Global ID {speaker_id}")
            
        except Exception as e:
            print(f"       ❌ Speaker ID分配失敗: {e}")
            continue
    
    return valid_segments, local_to_global_map


def compute_segment_embedding(
    audio_path: str,
    start: float,
    end: float,
    embedding_model,
    device: torch.device
) -> Optional[np.ndarray]:
    """為單個音頻片段計算embedding"""
    
    segment_dict = {
        "audio": audio_path,
        "start": start,
        "end": end
    }
    
    try:
        with torch.no_grad():
            embedding_output = embedding_model(segment_dict)
        
        # 處理不同類型的輸出
        if hasattr(embedding_output, 'cpu'):
            embedding_np = embedding_output.cpu().numpy()
        elif isinstance(embedding_output, dict):
            if 'embeddings' in embedding_output:
                embedding_np = embedding_output['embeddings']
            elif 'embedding' in embedding_output:
                embedding_np = embedding_output['embedding']
            else:
                embedding_np = list(embedding_output.values())[0]
            
            if hasattr(embedding_np, 'cpu'):
                embedding_np = embedding_np.cpu().numpy()
        else:
            embedding_np = embedding_output
        
        # 確保是1D
        if embedding_np.ndim > 1:
            embedding_np = embedding_np.flatten()
        
        return embedding_np
        
    except Exception as e:
        print(f"     ⚠️ Embedding計算失敗: {e}")
        return None