#!/usr/bin/env python3
"""
混合分段模式 - 結合 Pyannote Diarization 與字幕時間點
- 使用 diarization 提供精確的說話人變化點
- 使用字幕時間軸確保不會遺漏任何字幕內容
- 優先保證字幕完整性，同時利用 diarization 的說話人辨識精度
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from tqdm import tqdm
import librosa
from pyannote.core import Annotation, Segment

def segment_by_hybrid_approach(
    diarization: Annotation,
    subtitles: List[Tuple[float, str]],
    audio_path: str,
    embedding_model,
    device: torch.device,
    db,
    episode_num: int,
    min_duration: float = 1.0,
    max_duration: float = 15.0,
    similarity_threshold: float = 0.30,
    voice_activity_threshold: float = 0.1
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """
    混合分段方法：結合 diarization 的說話人變化點與字幕時間軸
    
    策略：
    1. 以字幕時間點為主要分段基礎（確保不遺漏內容）
    2. 在字幕段落內部使用 diarization 提供的說話人標籤
    3. 合併相鄰的同說話人片段
    4. 最終分配全域 speaker ID
    """
    
    if not subtitles:
        print("   ❌ 沒有字幕資料")
        return [], {}
    
    if not diarization:
        print("   ❌ 沒有 diarization 結果")
        return [], {}
    
    print(f"   🎯 混合分段模式：{len(subtitles)} 句字幕 + diarization 說話人資訊")
    
    # Step 1: 建立字幕導向的初始分段，並加上 diarization 的說話人標籤
    subtitle_segments_with_speakers = create_subtitle_segments_with_diarization(
        subtitles, diarization, audio_path, voice_activity_threshold
    )
    
    print(f"   ✂️ 建立了 {len(subtitle_segments_with_speakers)} 個混合分段")
    
    # Step 2: 為每個分段提取 embedding
    segments_with_embeddings = extract_embeddings_for_hybrid_segments(
        subtitle_segments_with_speakers, audio_path, embedding_model, device
    )
    
    # 統計 embedding 提取成功率
    successful_embeddings = sum(1 for seg in segments_with_embeddings if seg.get('embedding_extracted', False))
    print(f"   🧬 提取了 {len(segments_with_embeddings)} 個 embedding (成功: {successful_embeddings}, 失敗: {len(segments_with_embeddings) - successful_embeddings})")
    
    # Step 3: 基於說話人標籤和 embedding 相似度合併相鄰片段
    merged_segments = merge_segments_by_speaker_and_similarity(
        segments_with_embeddings, similarity_threshold, max_duration
    )
    print(f"   🔗 合併後剩餘 {len(merged_segments)} 個段落")
    
    # Step 4: 過濾長度並分配 global speaker ID
    final_segments, local_to_global_map = assign_global_speaker_ids_hybrid(
        merged_segments, db, episode_num, min_duration, max_duration, similarity_threshold
    )
    
    print(f"   ✅ 最終產生 {len(final_segments)} 個有效段落")
    return final_segments, local_to_global_map


def create_subtitle_segments_with_diarization(
    subtitles: List[Tuple[float, str]],
    diarization: Annotation,
    audio_path: str,
    voice_activity_threshold: float = 0.1
) -> List[Dict]:
    """
    結合字幕時間點和 diarization 說話人標籤創建混合分段
    
    邏輯：
    1. 以字幕為主要時間軸
    2. 對每個字幕時間點，查找 diarization 中的主要說話人
    3. 保持字幕完整性的同時加入說話人資訊
    """
    print("   🔊 Loading audio for hybrid segmentation...")
    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
        audio_duration = waveform.shape[0] / sr
    except Exception as e:
        print(f"   ❌ Error loading audio: {e}")
        audio_duration = 3600.0  # 假設最長 1 小時
    
    segments = []
    print("   🎭 結合字幕與 diarization 說話人資訊...")
    
    filtered_count = 0  # 統計被過濾的片段數量
    
    for i, (start_time, text) in enumerate(tqdm(subtitles, desc="   Processing subtitles", unit="line", ncols=80)):
        # 計算結束時間 (你的原始邏輯是正確的)
        if i < len(subtitles) - 1:
            end_time = subtitles[i + 1][0]  # 下一段開始時間 = 這段結束時間
        else:
            end_time = audio_duration
        
        # 限制單個段落最大長度
        end_time = min(end_time, start_time + 20.0)
        
        if end_time <= start_time:
            filtered_count += 1
            continue
        
        # 簡化邏輯：直接用字幕時間，只做基本檢查
        if end_time > start_time + 0.5:  # 至少0.5秒長度
            # 在這個時間範圍內找到主要的說話人
            segment_range = Segment(start_time, end_time)
            dominant_speaker = get_dominant_speaker_in_range(diarization, segment_range)
            
            # 只要有說話人就保留（移除VAD複雜檢查）
            if dominant_speaker is not None:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'diarization_speaker': dominant_speaker,
                    'original_index': i,
                    'duration': end_time - start_time,
                    'overlap_ratio': 1.0  # 簡化：假設都是有效語音
                })
            else:
                filtered_count += 1
        else:
            filtered_count += 1
    
    print(f"   📊 過濾統計: 原始字幕 {len(subtitles)} 行, 保留 {len(segments)} 行, 過濾 {filtered_count} 行 (無語音活動)")
    return segments


def find_actual_speech_range_from_start_time(
    diarization: Annotation,
    subtitle_start_time: float,
    max_search_duration: float = 10.0
) -> Optional[Tuple[float, float, str]]:
    """
    從字幕開始時間點找到對應的實際語音範圍
    返回: (語音開始時間, 語音結束時間, 說話者) 或 None
    
    策略：
    1. 從字幕開始時間點開始搜尋
    2. 找到第一個包含該時間點的語音段
    3. 如果沒有直接包含，找距離最近的語音段（在搜尋範圍內）
    """
    best_match = None
    min_distance = float('inf')
    
    # 搜尋範圍
    search_start = subtitle_start_time
    search_end = subtitle_start_time + max_search_duration
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speech_start = turn.start
        speech_end = turn.end
        
        # 情況1：字幕時間點在語音段內 (最佳匹配)
        if speech_start <= subtitle_start_time <= speech_end:
            return (speech_start, speech_end, speaker)
        
        # 情況2：語音段在搜尋範圍內，計算距離
        if (speech_start <= search_end and speech_end >= search_start):
            # 計算字幕開始時間與語音段的距離
            if subtitle_start_time < speech_start:
                distance = speech_start - subtitle_start_time  # 語音在後
            elif subtitle_start_time > speech_end:
                distance = subtitle_start_time - speech_end     # 語音在前
            else:
                distance = 0  # 重疊（已在情況1處理）
            
            # 更新最佳匹配
            if distance < min_distance:
                min_distance = distance
                best_match = (speech_start, speech_end, speaker)
    
    # 如果找到合理距離內的匹配（2秒內）
    if best_match and min_distance <= 2.0:
        return best_match
    
    return None


def calculate_diarization_overlap(diarization: Annotation, segment: Segment) -> float:
    """
    計算字幕片段與 diarization 結果的重疊比例
    返回值為 0.0-1.0，表示有多少比例的時間有語音活動
    """
    total_overlap_duration = 0.0
    segment_duration = segment.duration
    
    if segment_duration <= 0:
        return 0.0
    
    # 遍歷所有 diarization 片段，計算重疊時間
    for turn, _, _ in diarization.itertracks(yield_label=True):
        overlap = segment & turn
        if overlap:
            total_overlap_duration += overlap.duration
    
    # 返回重疊比例
    overlap_ratio = total_overlap_duration / segment_duration
    return min(1.0, overlap_ratio)  # 限制在 1.0 以內


def get_dominant_speaker_in_range(diarization: Annotation, segment: Segment) -> Optional[str]:
    """
    在給定時間範圍內找到說話時間最長的說話人
    """
    speaker_durations = {}
    
    # 遍歷 diarization 中所有與此段落重疊的部分
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 計算重疊部分
        overlap = segment & turn
        if overlap:
            duration = overlap.duration
            if speaker in speaker_durations:
                speaker_durations[speaker] += duration
            else:
                speaker_durations[speaker] = duration
    
    if not speaker_durations:
        return None
    
    # 回傳說話時間最長的說話人
    return max(speaker_durations.items(), key=lambda x: x[1])[0]


def extract_embeddings_for_hybrid_segments(
    segments: List[Dict],
    audio_path: str,
    embedding_model,
    device: torch.device
) -> List[Dict]:
    """為混合分段提取 speaker embedding"""
    
    segments_with_embeddings = []
    
    if not segments:
        return []

    try:
        print(f"   🔊 Loading audio file for embedding: {audio_path}")
        waveform, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"   ❌ Error: Could not load audio file {audio_path}. Error: {e}")
        for segment in segments:
            segment['embedding'] = np.zeros(512)
            segment['embedding_extracted'] = False
            segments_with_embeddings.append(segment)
        return segments_with_embeddings

    print(f"   Extracting embeddings for {len(segments)} hybrid segments...")
    for segment in tqdm(segments, desc="   Extracting embeddings", unit="seg", ncols=80):
        start, end = segment['start'], segment['end']
        
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_waveform = waveform[start_sample:end_sample]

        embedding_extracted = False
        final_embedding = np.zeros(512)

        if segment_waveform.size > sr * 0.3:  # 需要至少 0.3s 的音頻
            try:
                segment_tensor = torch.from_numpy(segment_waveform).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding_output = embedding_model(segment_tensor)
                
                if hasattr(embedding_output, 'cpu'):
                    final_embedding = embedding_output.cpu().numpy()[0].flatten()
                else:
                    final_embedding = np.array(embedding_output)[0].flatten()
                embedding_extracted = True

            except Exception as e:
                print(f"     ❌ Error extracting embedding for segment [{start:.1f}-{end:.1f}s]: {e}")
                embedding_extracted = False
        else:
            embedding_extracted = False

        segment['embedding'] = final_embedding
        segment['embedding_extracted'] = embedding_extracted
        segments_with_embeddings.append(segment)
    
    return segments_with_embeddings


def merge_segments_by_speaker_and_similarity(
    segments: List[Dict],
    similarity_threshold: float,
    max_duration: float
) -> List[Dict]:
    """
    基於 diarization 說話人標籤和 embedding 相似度合併相鄰片段
    
    合併條件：
    1. 相鄰片段
    2. 相同的 diarization 說話人標籤（如果有）
    3. Embedding 相似度高於閾值
    4. 合併後長度不超過最大限制
    """
    
    if len(segments) <= 1:
        return segments
    
    merged_segments = []
    current_segment = segments[0].copy()
    merge_count = 0
    
    print(f"   🔍 開始混合分段合併 (相似度閾值: {similarity_threshold})")
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        
        # 檢查是否可以合併
        can_merge = False
        similarity = 0.0
        
        # 條件 1: diarization 說話人標籤相同（如果有）
        speaker_match = (
            current_segment.get('diarization_speaker') == next_segment.get('diarization_speaker')
            and current_segment.get('diarization_speaker') is not None
        )
        
        # 條件 2: embedding 相似度
        curr_has_embedding = current_segment.get('embedding_extracted', False)
        next_has_embedding = next_segment.get('embedding_extracted', False)
        
        embedding_similar = False
        if curr_has_embedding and next_has_embedding:
            emb1 = current_segment['embedding']
            emb2 = next_segment['embedding']
            
            try:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                embedding_similar = similarity > similarity_threshold
            except Exception as e:
                print(f"     ⚠️ 相似度計算失敗: {e}")
                embedding_similar = False
        
        # 條件 3: 時間長度限制
        merged_duration = next_segment['end'] - current_segment['start']
        duration_ok = merged_duration <= max_duration
        
        # 合併決策：說話人標籤相同 AND embedding 相似 AND 長度合規
        if speaker_match and embedding_similar and duration_ok:
            can_merge = True
        elif not speaker_match and embedding_similar and duration_ok:
            # 如果沒有說話人標籤資訊，只靠 embedding
            can_merge = True
        
        # 顯示合併決策資訊
        speaker_info = f"diar_speaker: {current_segment.get('diarization_speaker', 'N/A')} vs {next_segment.get('diarization_speaker', 'N/A')}"
        similarity_info = f"embedding: {similarity:.3f} (>={similarity_threshold:.2f})" if curr_has_embedding and next_has_embedding else "embedding: N/A"
        
        print(f"     🔍 比較段落 {current_segment['original_index']+1} vs {next_segment['original_index']+1}: "
              f"{speaker_info}, {similarity_info}, "
              f"合併長度={merged_duration:.1f}s → {'✅合併' if can_merge else '❌分開'}")
        
        if can_merge:
            # 合併到 current_segment
            current_segment['end'] = next_segment['end']
            current_segment['duration'] = current_segment['end'] - current_segment['start']
            current_segment['text'] += " " + next_segment['text']
            
            # 使用平均 embedding
            if (curr_has_embedding and next_has_embedding):
                current_segment['embedding'] = (current_segment['embedding'] + next_segment['embedding']) / 2
            
            merge_count += 1
            print(f"     ✅ 合併成功: [{current_segment['start']:.1f}-{current_segment['end']:.1f}s] '{current_segment['text'][:50]}...'")
        else:
            # 無法合併，保存 current 並開始新的
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()
    
    # 添加最後一個段落
    merged_segments.append(current_segment)
    
    print(f"   📊 混合合併統計: 執行了 {merge_count} 次合併,最終 {len(merged_segments)} 個段落")
    return merged_segments


def assign_global_speaker_ids_hybrid(
    segments: List[Dict],
    db,
    episode_num: int,
    min_duration: float,
    max_duration: float,
    similarity_threshold: float
) -> Tuple[List[Tuple[float, float, int]], Dict[str, int]]:
    """為混合分段的每個段落分配 global speaker ID 並過濾長度"""
    
    valid_segments = []
    local_to_global_map = {}
    
    print(f"   🎯 分配 Global Speaker IDs (混合模式，相似度閾值: {similarity_threshold})")
    
    for i, segment in enumerate(segments):
        start, end = segment['start'], segment['end']
        duration = end - start
        diar_speaker = segment.get('diarization_speaker', 'unknown')
        
        print(f"     處理段落 #{i+1}: [{start:.1f}-{end:.1f}s] ({duration:.1f}s) diarization={diar_speaker}")
        
        # 過濾長度
        if not (min_duration <= duration <= max_duration):
            print(f"       ❌ 長度不合格 (需要 {min_duration}-{max_duration}s)")
            continue
        
        # 檢查是否有有效 embedding
        if not segment.get('embedding_extracted', False):
            print(f"       ❌ 無有效 embedding")
            continue
        
        embedding = segment['embedding']
        local_label = f"hybrid_seg_{i}_{diar_speaker}"
        
        # 在資料庫中查找或註冊 speaker
        try:
            print(f"       🔍 呼叫 find_similar_speaker，閾值={similarity_threshold}")
            
            # 檢查是否啟用embedding更新
            update_embeddings = os.getenv("UPDATE_SPEAKER_EMBEDDINGS", "false").lower() == "true"
            update_weight = float(os.getenv("EMBEDDING_UPDATE_WEIGHT", "1.0"))
            
            speaker_id, similarity = db.find_similar_speaker(embedding, similarity_threshold,
                                                           update_embeddings, update_weight)
            print(f"       🔍 find_similar_speaker 返回: speaker_id={speaker_id}, similarity={similarity}")
            
            if speaker_id is not None:
                print(f"       🔍 匹配現有 Global Speaker: {speaker_id} (相似度: {similarity:.3f})")
                db.update_speaker_episode(speaker_id, episode_num, local_label, 1)
            else:
                speaker_id = db.add_speaker(embedding, episode_num, local_label, 1)
                print(f"       ✨ 註冊新 Global Speaker: {speaker_id}")
            
            # 記錄 mapping 並添加到結果
            local_to_global_map[local_label] = speaker_id
            valid_segments.append((start, end, speaker_id))
            
            print(f"       ✅ 段落已保存: Global ID {speaker_id}")
            
        except Exception as e:
            print(f"       ❌ Speaker ID 分配失敗: {e}")
            continue
    
    return valid_segments, local_to_global_map