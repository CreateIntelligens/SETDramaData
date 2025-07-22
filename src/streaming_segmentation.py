#!/usr/bin/env python3
"""
Streaming-style speaker segmentation
連續決策的speaker分段，不做跳躍合併
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pyannote.core import Annotation, Segment
import torch


def segment_by_streaming_decision(
    diarization: Annotation,
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
    按時間順序連續決策切分，即時註冊新speaker
    返回: (segments_list, local_to_global_mapping)
    """
    
    # 按時間排序所有變化點
    timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        timeline.append((turn.start, turn.end, speaker))
    
    timeline.sort(key=lambda x: x[0])
    print(f"   📅 Timeline: {len(timeline)} speaker changes")
    
    segments = []
    current_segment = None
    local_to_global_map = {}
    
    def get_or_register_speaker(local_label: str, start_time: float, end_time: float) -> int:
        """即時獲取或註冊speaker的global ID"""
        if local_label in local_to_global_map:
            return local_to_global_map[local_label]
        
        # 提取這段的embedding
        print(f"     🧬 Extracting embedding for new speaker {local_label}")
        segment_dict = {
            "audio": audio_path,
            "start": start_time,
            "end": end_time
        }
        
        try:
            with torch.no_grad():
                embedding = embedding_model(segment_dict)
            
            # Handle different types of embedding outputs
            if hasattr(embedding, 'cpu'):
                embedding_np = embedding.cpu().numpy()
            elif isinstance(embedding, dict):
                # Handle case where embedding is returned as dict
                if 'embeddings' in embedding:
                    embedding_np = embedding['embeddings']
                elif 'embedding' in embedding:
                    embedding_np = embedding['embedding']
                else:
                    # Take the first value if it's a dict
                    embedding_np = list(embedding.values())[0]
                
                # Convert to numpy if still tensor
                if hasattr(embedding_np, 'cpu'):
                    embedding_np = embedding_np.cpu().numpy()
            else:
                embedding_np = embedding
            
            print(f"     📊 Embedding shape: {embedding_np.shape}, Mean: {np.mean(embedding_np):.4f}")
            
            # Debug: Check embedding dimensions
            if embedding_np.ndim == 1:
                print(f"     ✓ Valid 1D embedding: {embedding_np.shape[0]} features")
            else:
                print(f"     ⚠️ Unexpected embedding shape: {embedding_np.shape}, reshaping to 1D")
                embedding_np = embedding_np.flatten()
            
            # 立即在資料庫中查找或註冊
            speaker_id, similarity = db.find_similar_speaker(embedding_np, similarity_threshold)
            
            if speaker_id is not None:
                print(f"     🔍 Matched existing Global Speaker ID: {speaker_id} (Similarity: {similarity:.3f})")
                db.update_speaker_episode(speaker_id, episode_num, local_label, 1)
            else:
                speaker_id = db.add_speaker(embedding_np, episode_num, local_label, 1)
                print(f"     ✨ Registered new Global Speaker ID: {speaker_id}")
            
            local_to_global_map[local_label] = speaker_id
            return speaker_id
            
        except Exception as e:
            print(f"     ⚠️ Embedding extraction failed: {e}")
            # fallback: 直接註冊沒有embedding的speaker
            speaker_id = db.add_speaker(np.zeros(512), episode_num, local_label, 1)
            local_to_global_map[local_label] = speaker_id
            return speaker_id
    
    for i, (start, end, speaker_label) in enumerate(timeline):
        print(f"   🎯 Processing {speaker_label} at {start:.1f}-{end:.1f}s")
        
        # 立即獲取這個speaker的global ID
        global_speaker_id = get_or_register_speaker(speaker_label, start, end)
        
        if current_segment is None:
            # 第一段，直接開始
            current_segment = {
                'start': start,
                'end': end, 
                'speaker_label': speaker_label,
                'global_id': global_speaker_id,
                'segments_count': 1
            }
            print(f"     ✨ Starting new segment with Global ID {global_speaker_id}")
            continue
        
        # 檢查是否與當前段連續
        is_continuous = abs(current_segment['end'] - start) < 0.1  # 容忍0.1秒誤差
        is_same_global_speaker = current_segment['global_id'] == global_speaker_id
        
        # 檢查合併後是否超過最大長度
        merged_duration = end - current_segment['start']
        within_max_duration = merged_duration <= max_duration
        
        if is_continuous and is_same_global_speaker and within_max_duration:
            # 條件：連續 + 同一global speaker + 不超長 → 合併
            print(f"     ✅ Merging with Global ID {global_speaker_id} ({merged_duration:.1f}s)")
            current_segment['end'] = end
            current_segment['segments_count'] += 1
            print(f"     📈 Extended to {current_segment['end']:.1f}s ({current_segment['segments_count']} parts)")
            continue
        
        # 不能合併，結束當前segment，開始新的
        if current_segment['end'] - current_segment['start'] >= min_duration:
            segments.append((
                current_segment['start'], 
                current_segment['end'], 
                current_segment['global_id']  # 使用global ID
            ))
            print(f"     💾 Saved segment: Global ID {current_segment['global_id']} ({current_segment['end'] - current_segment['start']:.1f}s)")
        else:
            print(f"     🗑️ Discarded short segment: {current_segment['end'] - current_segment['start']:.1f}s")
        
        # 開始新segment
        current_segment = {
            'start': start,
            'end': end,
            'speaker_label': speaker_label,
            'global_id': global_speaker_id,
            'segments_count': 1
        }
        print(f"     🆕 New segment: Global ID {global_speaker_id}")
    
    # 處理最後一段
    if current_segment and (current_segment['end'] - current_segment['start']) >= min_duration:
        segments.append((
            current_segment['start'], 
            current_segment['end'], 
            current_segment['global_id']
        ))
        print(f"   💾 Final segment: Global ID {current_segment['global_id']} ({current_segment['end'] - current_segment['start']:.1f}s)")
    
    print(f"   ✅ Created {len(segments)} final segments")
    return segments, local_to_global_map


def verify_continuous_segments(
    audio_path: str,
    seg1_start: float, seg1_end: float,
    seg2_start: float, seg2_end: float,
    embedding_model,
    device: torch.device,
    threshold: float = 0.85
) -> bool:
    """
    驗證兩個連續segments是否為同一speaker
    """
    try:
        embeddings = []
        
        # 取每段的中間部分來比較
        for start, end in [(seg1_start, seg1_end), (seg2_start, seg2_end)]:
            if end - start < 0.5:  # 太短的段落
                return False
            
            # 取中間80%的部分
            duration = end - start
            margin = duration * 0.1
            safe_start = start + margin
            safe_end = end - margin
            
            segment_dict = {
                "audio": audio_path,
                "start": safe_start,
                "end": safe_end
            }
            
            try:
                with torch.no_grad():
                    embedding = embedding_model(segment_dict)
                
                if hasattr(embedding, 'cpu'):
                    embeddings.append(embedding.cpu().numpy())
                else:
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"     ⚠️ Embedding extraction failed: {e}")
                return True  # 如果提取失敗，保守地允許合併
        
        if len(embeddings) == 2:
            # 計算相似度
            similarity_tensor = torch.nn.functional.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            )
            similarity = similarity_tensor.item() if similarity_tensor.numel() == 1 else similarity_tensor[0].item()
            
            print(f"     📊 Similarity: {similarity:.3f}")
            return similarity > threshold
            
        return False
        
    except Exception as e:
        print(f"     ⚠️ Verification failed: {e}")
        return True  # 出錯時保持合併