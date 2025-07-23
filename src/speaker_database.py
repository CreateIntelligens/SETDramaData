#!/usr/bin/env python3
"""
Speaker Database Management Module
Manages speaker embeddings and identification using SQLite
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from datetime import datetime


class SpeakerDatabase:
    """SQLite-based speaker database for embedding storage and matching"""
    
    def __init__(self, db_path: str = "speakers.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the speaker database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create speakers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    episode_count INTEGER DEFAULT 1,
                    segment_count INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            
            # Create episodes table for tracking which speakers appear in which episodes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS speaker_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    speaker_id INTEGER,
                    episode_num INTEGER,
                    local_label TEXT,
                    segment_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speaker_id) REFERENCES speakers (speaker_id),
                    UNIQUE(speaker_id, episode_num)
                )
            """)
            
            # Create processing state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_speaker_episodes ON speaker_episodes(speaker_id, episode_num)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes ON speaker_episodes(episode_num)")
            
            conn.commit()
            
        print(f"🗄️ Speaker database initialized: {self.db_path}")
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes for storage"""
        return embedding.tobytes()
    
    def _deserialize_embedding(self, data: bytes, dim: int) -> np.ndarray:
        """Convert bytes back to numpy array"""
        # Always deserialize as 1D array first
        flat_array = np.frombuffer(data, dtype=np.float32)
        # For pyannote embeddings, we typically want 1D arrays
        return flat_array
    
    def add_speaker(self, embedding: np.ndarray, episode_num: int, local_label: str, segment_count: int = 0) -> int:
        """Add a new speaker to the database and return the speaker_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Ensure embedding is float32 for consistency
            embedding = embedding.astype(np.float32)
            # Store the total size for 1D embeddings (pyannote typically produces 1D)
            embedding_dim = embedding.size
            
            # Insert new speaker
            cursor.execute("""
                INSERT INTO speakers (embedding, embedding_dim, segment_count, notes)
                VALUES (?, ?, ?, ?)
            """, (
                self._serialize_embedding(embedding),
                embedding_dim,
                segment_count,
                f"First appeared in episode {episode_num} as {local_label}"
            ))
            
            speaker_id = cursor.lastrowid
            
            # Record episode appearance
            cursor.execute("""
                INSERT OR IGNORE INTO speaker_episodes 
                (speaker_id, episode_num, local_label, segment_count)
                VALUES (?, ?, ?, ?)
            """, (speaker_id, episode_num, local_label, segment_count))
            
            conn.commit()
            
        print(f"   ✨ New speaker registered: Global ID {speaker_id} (Episode {episode_num}, {segment_count} segments)")
        return speaker_id
    
    def find_similar_speaker(self, embedding: np.ndarray, similarity_threshold: float = 0.30, 
                           update_embedding: bool = False, update_weight: float = 1.0) -> Tuple[Optional[int], float]:
        """Find the most similar speaker in the database
        
        Args:
            embedding: Input embedding to match
            similarity_threshold: Minimum similarity threshold
            update_embedding: Whether to update matched speaker's embedding
            update_weight: Weight for the new embedding in update (default: 1.0)
            
        Returns:
            Tuple of (speaker_id, similarity) or (None, max_similarity)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all speakers with their embeddings
            cursor.execute("SELECT speaker_id, embedding, embedding_dim FROM speakers")
            speakers = cursor.fetchall()
            
            if not speakers:
                return None, 0.0
            
            # Ensure input embedding is float32
            embedding = embedding.astype(np.float32)
            current_embedding_tensor = torch.tensor(embedding).unsqueeze(0)
            
            max_similarity = 0.0
            best_speaker_id = None
            
            # Compare with each stored speaker
            for speaker_id, embedding_bytes, embedding_dim in speakers:
                stored_embedding = self._deserialize_embedding(embedding_bytes, embedding_dim)
                stored_tensor = torch.tensor(stored_embedding).unsqueeze(0)
                
                # Compute cosine similarity
                similarity_tensor = F.cosine_similarity(current_embedding_tensor, stored_tensor)
                try:
                    similarity = float(similarity_tensor.item())
                except (RuntimeError, ValueError):
                    # Handle multi-dimensional tensor
                    similarity_np = similarity_tensor.squeeze().cpu().numpy()
                    similarity = float(similarity_np.flatten()[0])
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_speaker_id = speaker_id
            
            if max_similarity > similarity_threshold:
                print(f"   🔍 Match found! Speaker is likely Global Speaker ID: {best_speaker_id} (Similarity: {max_similarity:.3f})")
                
                # Update embedding if requested
                if update_embedding and best_speaker_id is not None:
                    self.update_speaker_embedding(best_speaker_id, embedding, update_weight)
                
                return best_speaker_id, max_similarity
            
            return None, max_similarity
    
    def update_speaker_embedding(self, speaker_id: int, new_embedding: np.ndarray, new_weight: float = 1.0) -> bool:
        """Update speaker embedding using weighted average
        
        Args:
            speaker_id: Target speaker ID
            new_embedding: New embedding to merge
            new_weight: Weight for the new embedding (default: 1.0)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current speaker data
            cursor.execute("""
                SELECT embedding, embedding_dim, segment_count 
                FROM speakers WHERE speaker_id = ?
            """, (speaker_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"   ❌ Speaker {speaker_id} not found for embedding update")
                return False
                
            old_embedding_bytes, embedding_dim, segment_count = result
            old_embedding = self._deserialize_embedding(old_embedding_bytes, embedding_dim)
            
            # Calculate weights
            old_weight = max(1.0, float(segment_count))  # Minimum weight of 1
            total_weight = old_weight + new_weight
            
            # Compute weighted average
            old_embedding = old_embedding.astype(np.float32)
            new_embedding = new_embedding.astype(np.float32)
            
            updated_embedding = (old_weight * old_embedding + new_weight * new_embedding) / total_weight
            
            # Normalize to unit length (important for cosine similarity)
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            
            # Serialize and update in database
            updated_embedding_bytes = self._serialize_embedding(updated_embedding)
            
            cursor.execute("""
                UPDATE speakers 
                SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ?
            """, (updated_embedding_bytes, speaker_id))
            
            conn.commit()
            
            print(f"   🔄 Updated embedding for Speaker {speaker_id} (weights: {old_weight:.1f} + {new_weight:.1f})")
            return True
    
    def update_speaker_episode(self, speaker_id: int, episode_num: int, local_label: str, segment_count: int = 0):
        """Record that a speaker appeared in an episode"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update or insert episode record
            cursor.execute("""
                INSERT OR REPLACE INTO speaker_episodes 
                (speaker_id, episode_num, local_label, segment_count)
                VALUES (?, ?, ?, ?)
            """, (speaker_id, episode_num, local_label, segment_count))
            
            # Update speaker's episode count
            cursor.execute("""
                UPDATE speakers 
                SET episode_count = (
                    SELECT COUNT(DISTINCT episode_num) 
                    FROM speaker_episodes 
                    WHERE speaker_id = ?
                ),
                segment_count = segment_count + ?,
                updated_at = CURRENT_TIMESTAMP
                WHERE speaker_id = ?
            """, (speaker_id, segment_count, speaker_id))
            
            conn.commit()
    
    def get_speaker_info(self, speaker_id: int) -> Optional[Dict]:
        """Get detailed information about a speaker"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get speaker details
            cursor.execute("""
                SELECT speaker_id, embedding_dim, created_at, updated_at, 
                       episode_count, segment_count, notes
                FROM speakers WHERE speaker_id = ?
            """, (speaker_id,))
            
            speaker = cursor.fetchone()
            if not speaker:
                return None
            
            # Get episodes this speaker appeared in
            cursor.execute("""
                SELECT episode_num, local_label, segment_count, created_at
                FROM speaker_episodes 
                WHERE speaker_id = ?
                ORDER BY episode_num
            """, (speaker_id,))
            
            episodes = cursor.fetchall()
            
            return {
                'speaker_id': speaker[0],
                'embedding_dim': speaker[1],
                'created_at': speaker[2],
                'updated_at': speaker[3],
                'episode_count': speaker[4],
                'total_segments': speaker[5],
                'notes': speaker[6],
                'episodes': [
                    {
                        'episode_num': ep[0],
                        'local_label': ep[1],
                        'segment_count': ep[2],
                        'created_at': ep[3]
                    }
                    for ep in episodes
                ]
            }
    
    def list_all_speakers(self) -> List[Dict]:
        """Get summary of all speakers"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT s.speaker_id, s.episode_count, s.segment_count, 
                       s.created_at, s.updated_at,
                       GROUP_CONCAT(se.episode_num) as episodes
                FROM speakers s
                LEFT JOIN speaker_episodes se ON s.speaker_id = se.speaker_id
                GROUP BY s.speaker_id
                ORDER BY s.speaker_id
            """)
            
            speakers = cursor.fetchall()
            
            return [
                {
                    'speaker_id': speaker[0],
                    'episode_count': speaker[1],
                    'segment_count': speaker[2],
                    'created_at': speaker[3],
                    'updated_at': speaker[4],
                    'episodes': speaker[5].split(',') if speaker[5] else []
                }
                for speaker in speakers
            ]
    
    def get_processed_episodes(self) -> List[int]:
        """Get list of processed episodes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT value FROM processing_state WHERE key = 'processed_episodes'
            """)
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            return []
    
    def mark_episode_processed(self, episode_num: int):
        """Mark an episode as processed"""
        processed = self.get_processed_episodes()
        if episode_num not in processed:
            processed.append(episode_num)
            processed.sort()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO processing_state (key, value, updated_at)
                    VALUES ('processed_episodes', ?, CURRENT_TIMESTAMP)
                """, (json.dumps(processed),))
                conn.commit()
    
    def get_episode_speaker_mapping(self, episode_num: int) -> Dict[str, int]:
        """Get local to global speaker mapping for an episode"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT local_label, speaker_id
                FROM speaker_episodes
                WHERE episode_num = ?
            """, (episode_num,))
            
            return {local_label: speaker_id for local_label, speaker_id in cursor.fetchall()}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total speakers
            cursor.execute("SELECT COUNT(*) FROM speakers")
            total_speakers = cursor.fetchone()[0]
            
            # Total episodes processed
            cursor.execute("SELECT COUNT(DISTINCT episode_num) FROM speaker_episodes")
            total_episodes = cursor.fetchone()[0]
            
            # Total segments
            cursor.execute("SELECT SUM(segment_count) FROM speakers")
            total_segments = cursor.fetchone()[0] or 0
            
            # Database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            return {
                'total_speakers': total_speakers,
                'total_episodes': total_episodes,
                'total_segments': total_segments,
                'database_size_bytes': db_size,
                'database_size_mb': round(db_size / 1024 / 1024, 2),
                'database_path': str(self.db_path)
            }
    
    def export_to_json(self, output_path: str):
        """Export database to JSON for backup"""
        data = {
            'speakers': [],
            'processed_episodes': self.get_processed_episodes(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        for speaker_id in [s['speaker_id'] for s in self.list_all_speakers()]:
            speaker_info = self.get_speaker_info(speaker_id)
            if speaker_info:
                # Convert embedding to list for JSON serialization
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT embedding, embedding_dim FROM speakers WHERE speaker_id = ?", (speaker_id,))
                    embedding_bytes, dim = cursor.fetchone()
                    embedding = self._deserialize_embedding(embedding_bytes, dim)
                    speaker_info['embedding'] = embedding.tolist()
                
                data['speakers'].append(speaker_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Database exported to: {output_path}")


# Migration utility function
def migrate_from_json(json_path: str, db_path: str = "speakers.db"):
    """Migrate from old JSON format to SQLite database"""
    if not Path(json_path).exists():
        print(f"❌ JSON file not found: {json_path}")
        return
    
    db = SpeakerDatabase(db_path)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    print(f"🔄 Migrating from {json_path} to {db_path}...")
    
    # Migrate global speaker embeddings
    embeddings = old_data.get('global_speaker_embeddings', {})
    processed_episodes = old_data.get('processed_episodes', [])
    episode_mappings = old_data.get('episode_speaker_mapping', {})
    
    for speaker_id_str, embedding_list in embeddings.items():
        speaker_id = int(speaker_id_str)
        embedding = np.array(embedding_list, dtype=np.float32)
        
        # Find first episode this speaker appeared in
        first_episode = None
        for ep_str, mapping in episode_mappings.items():
            if speaker_id in mapping.values():
                first_episode = int(ep_str)
                break
        
        if first_episode is None:
            first_episode = processed_episodes[0] if processed_episodes else 1
        
        # Add speaker to database
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO speakers (speaker_id, embedding, embedding_dim, notes)
                VALUES (?, ?, ?, ?)
            """, (
                speaker_id,
                db._serialize_embedding(embedding),
                embedding.shape[0],
                f"Migrated from JSON (first seen in episode {first_episode})"
            ))
    
    # Migrate episode mappings
    for ep_str, mapping in episode_mappings.items():
        episode_num = int(ep_str)
        for local_label, speaker_id in mapping.items():
            db.update_speaker_episode(speaker_id, episode_num, local_label)
    
    # Mark episodes as processed
    for episode_num in processed_episodes:
        db.mark_episode_processed(episode_num)
    
    print(f"✅ Migration completed! Migrated {len(embeddings)} speakers and {len(processed_episodes)} episodes")
    
    # Backup original JSON
    backup_path = f"{json_path}.backup"
    Path(json_path).rename(backup_path)
    print(f"📦 Original JSON backed up to: {backup_path}")


if __name__ == "__main__":
    # Quick test/demo
    db = SpeakerDatabase("test_speakers.db")
    stats = db.get_database_stats()
    print("Database stats:", stats)