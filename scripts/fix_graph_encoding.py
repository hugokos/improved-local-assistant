#!/usr/bin/env python3
"""
Fix encoding issues in prebuilt graph JSON files.
"""

import json
import shutil
from pathlib import Path


def fix_json_encoding(file_path):
    """Fix encoding for a single JSON file."""
    try:
        # Read with multiple encoding attempts
        content = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()
                print(f"✅ Read {file_path.name} with {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"❌ Could not read {file_path.name} with any encoding")
            return False

        # Parse and re-save as UTF-8
        try:
            data = json.loads(content)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Fixed encoding for {file_path.name}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {file_path.name}: {e}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {file_path.name}: {e}")
        return False


def main():
    """Fix encoding for all JSON files in survivalist graph."""
    graph_dir = Path("./data/prebuilt_graphs/survivalist")

    if not graph_dir.exists():
        print(f"❌ Graph directory not found: {graph_dir}")
        return

    print(f"🔧 Fixing encoding for JSON files in {graph_dir}")

    # Create backup
    backup_dir = graph_dir.parent / "survivalist_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(graph_dir, backup_dir)
    print(f"📦 Created backup at {backup_dir}")

    # Fix each JSON file
    json_files = list(graph_dir.glob("*.json"))
    success_count = 0

    for json_file in json_files:
        if fix_json_encoding(json_file):
            success_count += 1

    print(f"\n🎉 Fixed {success_count}/{len(json_files)} JSON files")

    if success_count == len(json_files):
        print("✅ All files fixed successfully!")
    else:
        print("⚠️  Some files could not be fixed")


if __name__ == "__main__":
    main()
