"""
Migration Script: Move existing flat data into year-specific directories.

Moves data from:
  data/raw/*.htm           → data/raw/2025/
  data/extracted/*.txt     → data/extracted/2025/
  data/chunks/*.json       → data/chunks/2025/
  data/embeddings/*        → data/embeddings/2025/
  data/risk_profiles/*.json → data/risk_profiles/2025/
"""

import os
import shutil
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DIRS_TO_MIGRATE = ["raw", "extracted", "chunks", "embeddings", "risk_profiles"]
TARGET_YEAR = 2025

# File extensions to migrate per directory
EXTENSIONS = {
    "raw": ["*.htm", "*.html", "collection_metadata.json"],
    "extracted": ["*.txt", "extraction_metadata.json"],
    "chunks": ["*.json"],
    "embeddings": ["*.bin", "*.pkl", "*.npy", "*.json"],
    "risk_profiles": ["*.json"],
}


def migrate():
    """Move flat files into year-specific subdirectories."""
    print(f"{'='*60}")
    print(f"Migration: Moving flat data → FY{TARGET_YEAR} subdirectories")
    print(f"{'='*60}")
    
    for dir_name in DIRS_TO_MIGRATE:
        src_dir = os.path.join(DATA_DIR, dir_name)
        dst_dir = os.path.join(DATA_DIR, dir_name, str(TARGET_YEAR))
        
        if not os.path.exists(src_dir):
            print(f"\n  SKIP: {src_dir} does not exist")
            continue
            
        # Check if already migrated
        if os.path.exists(dst_dir) and os.listdir(dst_dir):
            print(f"\n  SKIP: {dst_dir} already exists and is not empty")
            continue
        
        # Find files to migrate (only files, not directories)
        files_to_move = []
        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)
            if os.path.isfile(item_path):
                files_to_move.append(item)
        
        if not files_to_move:
            print(f"\n  SKIP: {dir_name}/ — no files to migrate")
            continue
        
        # Create target directory
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"\n  Migrating {dir_name}/")
        for filename in files_to_move:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"    ✓ {filename}")
        
        print(f"    Moved {len(files_to_move)} files → {dir_name}/{TARGET_YEAR}/")
    
    print(f"\n{'='*60}")
    print("Migration complete!")
    print(f"{'='*60}")
    print("\nNote: Original files are kept in place for safety.")
    print("You can delete them manually after verifying the migration.")


if __name__ == "__main__":
    migrate()
