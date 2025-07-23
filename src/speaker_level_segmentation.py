#!/usr/bin/env python3
"""
說話人級別分段系統 - 兩階段說話人識別
階段1：單集內說話人整合 - 合併同說話人片段，提取說話人級別的 embedding
階段2：跨集說話人匹配 - 與全域資料庫比對，分配 Global Speaker ID
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from tqdm import tqdm
import librosa
import soundfile as sf
from pyannote.core import Annotation, Segment
from collections import defaultdict


def segment_by_speaker_level_approach(
    diarization: Annotation,
    subtitles: List[Tuple[float, str]],
    audio_path: str,
    embedding_model,
    device: torch.device,
    db,
    episode_num: int,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    similarity_threshold: float = 0.40,
    min_speaker_duration: float = 5.0
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """
    說話人級別分段方法：兩階段說話人識別
    
    階段1：單集內說話人整合
    1. 使用 diarization 結果識別說話人變化點
    2. 合併同說話人的所有片段
    3. 對每個說話人的完整音檔提取 embedding
    
    階段2：跨集說話人匹配
    1. 將說話人 embedding 與資料庫比對
    2. 分配 Global Speaker ID
    3. 根據字幕時間點生成最終分段
    """
    
    if not subtitles:
        print("   ❌ 沒有字幕資料")
        return [], {}
    
    if not diarization:
        print("   ❌ 沒有 diarization 結果")
        return [], {}
    
    print(f"   🎯 說話人級別分段模式：{len(subtitles)} 句字幕 + diarization 說話人資訊")
    
    # 階段1：單集內說話人整合
    print("   📊 階段1：單集內說話人整合")
    speaker_segments = extract_speaker_segments_from_diarization(diarization, audio_path)
    print(f"   ✂️ 從 diarization 提取了 {len(speaker_segments)} 個說話人")
    
    # 過濾太短的說話人
    valid_speakers = {speaker: segments for speaker, segments in speaker_segments.items() 
                     if calculate_total_duration(segments) >= min_speaker_duration}
    filtered_count = len(speaker_segments) - len(valid_speakers)
    if filtered_count > 0:
        print(f"   🗑️ 過濾了 {filtered_count} 個說話時間太短的說話人 (< {min_speaker_duration}s)")
    
    # 為每個說話人提取 embedding
    speaker_embeddings = extract_speaker_level_embeddings(
        valid_speakers, audio_path, embedding_model, device
    )
    print(f"   🧬 成功提取了 {len(speaker_embeddings)} 個說話人的 embedding")
    
    # 階段2：跨集說話人匹配
    print("   🔍 階段2：跨集說話人匹配")
    local_to_global_map = assign_global_speaker_ids_by_embedding(
        speaker_embeddings, valid_speakers, db, episode_num, similarity_threshold
    )
    
    # 生成最終分段（基於字幕時間點）
    print("   📝 生成最終分段")
    final_segments = generate_final_segments_with_subtitles(
        subtitles, diarization, local_to_global_map, min_duration, max_duration
    )
    
    print(f"   ✅ 最終產生 {len(final_segments)} 個有效段落")
    return final_segments, local_to_global_map


def extract_speaker_segments_from_diarization(
    diarization: Annotation, 
    audio_path: str
) -> Dict[str, List[Segment]]:
    """
    從 diarization 結果中提取每個說話人的所有片段
    返回：{speaker_label: [Segment1, Segment2, ...]}
    """
    speaker_segments = defaultdict(list)
    
    print("   🎭 從 diarization 提取說話人片段...")
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[speaker].append(turn)
    
    # 統計每個說話人的資訊
    for speaker, segments in speaker_segments.items():
        total_duration = sum(seg.duration for seg in segments)
        print(f"     {speaker}: {len(segments)} 個片段, 總時長 {total_duration:.1f}s")
    
    return dict(speaker_segments)


def calculate_total_duration(segments: List[Segment]) -> float:
    """計算片段列表的總時長"""
    return sum(seg.duration for seg in segments)


def extract_speaker_level_embeddings(
    speaker_segments: Dict[str, List[Segment]],
    audio_path: str,
    embedding_model,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    為每個說話人提取代表性的 embedding
    使用完整的說話段落而非短片段
    """
    print("   🔊 載入音檔進行 embedding 提取...")
    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"   ❌ 無法載入音檔: {e}")
        return {}
    
    speaker_embeddings = {}
    
    print("   🧬 為每個說話人提取 embedding...")
    for speaker, segments in tqdm(speaker_segments.items(), desc="   提取說話人 embedding", unit="speaker", ncols=80):
        # 合併所有屬於該說話人的音檔片段
        combined_audio = combine_speaker_audio_segments(waveform, sr, segments)
        
        if combined_audio.size < sr * 1.0:  # 至少需要1秒的音檔
            print(f"     ⚠️ {speaker}: 音檔太短 ({combined_audio.size/sr:.1f}s)，跳過")
            continue
        
        # 提取 embedding
        try:
            embedding = extract_embedding_from_audio(combined_audio, embedding_model, device)
            if embedding is not None:
                speaker_embeddings[speaker] = embedding
                print(f"     ✅ {speaker}: embedding 提取成功 ({combined_audio.size/sr:.1f}s 音檔)")
            else:
                print(f"     ❌ {speaker}: embedding 提取失敗")
        except Exception as e:
            print(f"     ❌ {speaker}: embedding 提取錯誤 - {e}")
    
    return speaker_embeddings


def combine_speaker_audio_segments(
    waveform: np.ndarray, 
    sr: int, 
    segments: List[Segment]
) -> np.ndarray:
    """
    合併同一說話人的所有音檔片段
    返回合併後的音檔數據
    """
    combined_segments = []
    
    for segment in segments:
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        
        # 確保索引在有效範圍內
        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)
        
        if end_sample > start_sample:
            segment_audio = waveform[start_sample:end_sample]
            combined_segments.append(segment_audio)
    
    if combined_segments:
        # 將所有片段串接起來
        return np.concatenate(combined_segments)
    else:
        return np.array([])


def extract_embedding_from_audio(
    audio_data: np.ndarray,
    embedding_model,
    device: torch.device
) -> Optional[np.ndarray]:
    """
    從音檔數據提取 embedding
    """
    try:
        # 轉換為 tensor
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding_output = embedding_model(audio_tensor)
        
        # 處理輸出
        if hasattr(embedding_output, 'cpu'):
            embedding = embedding_output.cpu().numpy()[0].flatten()
        else:
            embedding = np.array(embedding_output)[0].flatten()
        
        return embedding.astype(np.float32)
        
    except Exception as e:
        print(f"       ❌ Embedding 提取錯誤: {e}")
        return None


def assign_global_speaker_ids_by_embedding(
    speaker_embeddings: Dict[str, np.ndarray],
    speaker_segments: Dict[str, List[Segment]],
    db,
    episode_num: int,
    similarity_threshold: float
) -> Dict[str, int]:
    """
    基於 embedding 相似度分配 Global Speaker ID
    """
    local_to_global_map = {}
    
    print(f"   🎯 為 {len(speaker_embeddings)} 個說話人分配 Global Speaker ID")
    print(f"   🔍 相似度閾值: {similarity_threshold}")
    
    for local_speaker, embedding in speaker_embeddings.items():
        segments = speaker_segments[local_speaker]
        total_duration = calculate_total_duration(segments)
        segment_count = len(segments)
        
        print(f"     處理說話人 {local_speaker}: {segment_count} 個片段, 總時長 {total_duration:.1f}s")
        
        try:
            # 檢查是否啟用 embedding 更新
            update_embeddings = os.getenv("UPDATE_SPEAKER_EMBEDDINGS", "false").lower() == "true"
            update_weight = float(os.getenv("EMBEDDING_UPDATE_WEIGHT", "1.0"))
            
            # 在資料庫中查找相似說話人
            speaker_id, similarity = db.find_similar_speaker(
                embedding, similarity_threshold, update_embeddings, update_weight
            )
            
            if speaker_id is not None:
                print(f"       🔍 匹配到現有說話人: Global ID {speaker_id} (相似度: {similarity:.3f})")
                # 更新說話人在此集數的出現記錄
                db.update_speaker_episode(speaker_id, episode_num, local_speaker, segment_count)
            else:
                # 註冊新說話人
                speaker_id = db.add_speaker(embedding, episode_num, local_speaker, segment_count)
                print(f"       ✨ 註冊新說話人: Global ID {speaker_id}")
            
            local_to_global_map[local_speaker] = speaker_id
            
        except Exception as e:
            print(f"       ❌ 說話人 {local_speaker} ID 分配失敗: {e}")
            continue
    
    print(f"   📊 成功分配 {len(local_to_global_map)} 個說話人的 Global ID")
    return local_to_global_map


def generate_final_segments_with_subtitles(
    subtitles: List[Tuple[float, str]],
    diarization: Annotation,
    local_to_global_map: Dict[str, int],
    min_duration: float,
    max_duration: float
) -> List[Tuple[float, float, int]]:
    """
    基於字幕時間點生成最終分段，並分配 Global Speaker ID
    """
    final_segments = []
    
    print("   📝 基於字幕生成最終分段...")
    
    for i, (start_time, text) in enumerate(tqdm(subtitles, desc="   處理字幕", unit="line", ncols=80)):
        # 計算結束時間
        if i < len(subtitles) - 1:
            end_time = subtitles[i + 1][0]
        else:
            # 最後一句，估算結束時間
            end_time = start_time + min(10.0, max_duration)
        
        # 限制最大長度
        end_time = min(end_time, start_time + max_duration)
        duration = end_time - start_time
        
        # 過濾長度
        if duration < min_duration or duration > max_duration:
            continue
        
        # 找到這個時間範圍內的主要說話人
        segment_range = Segment(start_time, end_time)
        dominant_speaker = get_dominant_speaker_in_range(diarization, segment_range)
        
        if dominant_speaker and dominant_speaker in local_to_global_map:
            global_speaker_id = local_to_global_map[dominant_speaker]
            final_segments.append((start_time, end_time, global_speaker_id))
    
    return final_segments


def get_dominant_speaker_in_range(diarization: Annotation, segment: Segment) -> Optional[str]:
    """
    在給定時間範圍內找到說話時間最長的說話人
    """
    speaker_durations = {}
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = segment & turn
        if overlap:
            duration = overlap.duration
            if speaker in speaker_durations:
                speaker_durations[speaker] += duration
            else:
                speaker_durations[speaker] = duration
    
    if not speaker_durations:
        return None
    
    return max(speaker_durations.items(), key=lambda x: x[1])[0]
