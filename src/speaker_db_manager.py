#!/usr/bin/env python3
"""
Speaker Database Manager CLI Tool
Command-line interface for managing speaker database
"""

import argparse
import sys
from pathlib import Path
from speaker_database import SpeakerDatabase, migrate_from_json


def cmd_stats(args):
    """Show database statistics"""
    db = SpeakerDatabase(args.database)
    stats = db.get_database_stats()
    
    print("📊 Speaker Database Statistics")
    print("=" * 30)
    print(f"Database Path: {stats['database_path']}")
    print(f"Database Size: {stats['database_size_mb']} MB")
    print(f"Total Speakers: {stats['total_speakers']}")
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Total Segments: {stats['total_segments']}")
    
    if stats['total_speakers'] > 0:
        print(f"Avg Segments per Speaker: {stats['total_segments'] / stats['total_speakers']:.1f}")


def cmd_list_speakers(args):
    """List all speakers"""
    db = SpeakerDatabase(args.database)
    speakers = db.list_all_speakers()
    
    print("👥 All Speakers")
    print("=" * 50)
    print(f"{'ID':<4} {'Episodes':<12} {'Segments':<10} {'Created':<20}")
    print("-" * 50)
    
    for speaker in speakers:
        episodes_str = ','.join(speaker['episodes'][:5])  # Show first 5 episodes
        if len(speaker['episodes']) > 5:
            episodes_str += f"... (+{len(speaker['episodes'])-5})"
        
        print(f"{speaker['speaker_id']:<4} {episodes_str:<12} {speaker['segment_count']:<10} {speaker['created_at'][:19]:<20}")


def cmd_speaker_info(args):
    """Show detailed info for a specific speaker"""
    db = SpeakerDatabase(args.database)
    info = db.get_speaker_info(args.speaker_id)
    
    if not info:
        print(f"❌ Speaker {args.speaker_id} not found")
        return
    
    print(f"👤 Speaker {info['speaker_id']} Details")
    print("=" * 30)
    print(f"Created: {info['created_at']}")
    print(f"Updated: {info['updated_at']}")
    print(f"Episode Count: {info['episode_count']}")
    print(f"Total Segments: {info['total_segments']}")
    print(f"Embedding Dimension: {info['embedding_dim']}")
    
    if info['notes']:
        print(f"Notes: {info['notes']}")
    
    print("\n📺 Episode Appearances:")
    print(f"{'Episode':<8} {'Local Label':<15} {'Segments':<10} {'Date':<20}")
    print("-" * 55)
    
    for ep in info['episodes']:
        print(f"{ep['episode_num']:<8} {ep['local_label']:<15} {ep['segment_count']:<10} {ep['created_at'][:19]:<20}")


def cmd_episode_info(args):
    """Show speaker mapping for an episode"""
    db = SpeakerDatabase(args.database)
    mapping = db.get_episode_speaker_mapping(args.episode_num)
    processed_episodes = db.get_processed_episodes()
    
    print(f"📺 Episode {args.episode_num} Speaker Mapping")
    print("=" * 40)
    
    if args.episode_num not in processed_episodes:
        print("❌ Episode not processed yet")
        return
    
    if not mapping:
        print("❌ No speaker mapping found")
        return
    
    print(f"{'Local Label':<15} {'Global Speaker ID':<18}")
    print("-" * 35)
    
    for local_label, speaker_id in mapping.items():
        print(f"{local_label:<15} {speaker_id:<18}")


def cmd_export(args):
    """Export database to JSON"""
    db = SpeakerDatabase(args.database)
    db.export_to_json(args.output)
    print(f"✅ Database exported to {args.output}")


def cmd_migrate(args):
    """Migrate from JSON to SQLite"""
    if not Path(args.json_file).exists():
        print(f"❌ JSON file not found: {args.json_file}")
        return
    
    migrate_from_json(args.json_file, args.database)
    print("✅ Migration completed!")


def cmd_backup(args):
    """Create a backup of the database"""
    import shutil
    from datetime import datetime
    
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"❌ Database not found: {args.database}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"
    
    shutil.copy2(db_path, backup_path)
    print(f"✅ Database backed up to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="Speaker Database Manager")
    parser.add_argument("--database", "-d", default="data/speakers.db", 
                       help="Path to SQLite database (default: data/speakers.db)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")
    
    # List speakers command
    subparsers.add_parser("list", help="List all speakers")
    
    # Speaker info command
    speaker_parser = subparsers.add_parser("speaker", help="Show speaker details")
    speaker_parser.add_argument("speaker_id", type=int, help="Speaker ID")
    
    # Episode info command
    episode_parser = subparsers.add_parser("episode", help="Show episode speaker mapping")
    episode_parser.add_argument("episode_num", type=int, help="Episode number")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("output", help="Output JSON file path")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from JSON to SQLite")
    migrate_parser.add_argument("json_file", help="Input JSON file path")
    
    # Backup command
    subparsers.add_parser("backup", help="Create database backup")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    command_map = {
        "stats": cmd_stats,
        "list": cmd_list_speakers,
        "speaker": cmd_speaker_info,
        "episode": cmd_episode_info,
        "export": cmd_export,
        "migrate": cmd_migrate,
        "backup": cmd_backup
    }
    
    command_map[args.command](args)


if __name__ == "__main__":
    main()