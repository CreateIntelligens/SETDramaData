#!/usr/bin/env python3
"""
Database cleanup utilities for specific episodes
"""

import sys
import argparse
from pathlib import Path
from speaker_database import SpeakerDatabase


def remove_episodes_from_database(db_path: str, episodes: list):
    """Remove specific episodes from the SQLite database"""
    db = SpeakerDatabase(db_path)
    
    print(f"🗄️ 清理資料庫中的集數: {episodes}")
    
    # Get current state
    stats_before = db.get_database_stats()
    processed_episodes = db.get_processed_episodes()
    
    print(f"   清理前: {stats_before['total_speakers']} speakers, {stats_before['total_episodes']} episodes")
    
    # Remove episodes from processed list
    removed_episodes = []
    for episode in episodes:
        if episode in processed_episodes:
            processed_episodes.remove(episode)
            removed_episodes.append(episode)
    
    # Update processed episodes list
    if removed_episodes:
        import sqlite3
        import json
        
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processing_state (key, value, updated_at)
                VALUES ('processed_episodes', ?, CURRENT_TIMESTAMP)
            """, (json.dumps(processed_episodes),))
            
            # Remove episode records from speaker_episodes table
            for episode in removed_episodes:
                cursor.execute("""
                    DELETE FROM speaker_episodes WHERE episode_num = ?
                """, (episode,))
            
            conn.commit()
        
        print(f"   ✅ 已從資料庫移除 {len(removed_episodes)} 個集數")
        
        # Show updated stats
        stats_after = db.get_database_stats()
        print(f"   清理後: {stats_after['total_speakers']} speakers, {stats_after['total_episodes']} episodes")
    else:
        print(f"   ⚠️ 指定的集數不在資料庫中")


def show_episodes_from_database(db_path: str):
    """Show processed episodes from SQLite database"""
    if not Path(db_path).exists():
        print("❌ SQLite資料庫不存在")
        return
    
    db = SpeakerDatabase(db_path)
    processed_episodes = db.get_processed_episodes()
    
    if processed_episodes:
        print(f"🗄️ 資料庫中的已處理集數: {sorted(processed_episodes)}")
        print(f"   總計: {len(processed_episodes)} 集")
        
        # Show detailed speaker info per episode
        speakers = db.list_all_speakers()
        if speakers:
            print("\n📊 各集數詳情:")
            episode_details = {}
            
            for speaker in speakers:
                for episode_str in speaker['episodes']:
                    if episode_str:  # Check if not empty
                        try:
                            episode_num = int(episode_str)
                            if episode_num not in episode_details:
                                episode_details[episode_num] = {'speakers': 0, 'segments': 0}
                            episode_details[episode_num]['speakers'] += 1
                            episode_details[episode_num]['segments'] += speaker['segment_count']
                        except (ValueError, TypeError):
                            continue
            
            for episode in sorted(episode_details.keys()):
                details = episode_details[episode]
                print(f"   集數 {episode}: {details['speakers']} speakers, {details['segments']} segments")
    else:
        print("📋 資料庫中無已處理集數")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database cleanup utilities")
    parser.add_argument("--database", "-d", default="data/speakers.db", help="SQLite database path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Show episodes command
    subparsers.add_parser("show", help="Show processed episodes")
    
    # Remove episodes command
    remove_parser = subparsers.add_parser("remove", help="Remove specific episodes")
    remove_parser.add_argument("episodes", nargs="+", type=int, help="Episode numbers to remove")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "show":
        show_episodes_from_database(args.database)
    elif args.command == "remove":
        remove_episodes_from_database(args.database, args.episodes)