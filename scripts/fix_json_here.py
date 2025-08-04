#!/usr/bin/env python3
"""
Simple script to fix UTF-8 encoding issues in JSON files in the current directory.
This script can be copied to any directory with problematic JSON files.
"""

import os
import sys
import json
import shutil
from pathlib import Path

try:
    import chardet
except ImportError:
    print("Installing chardet...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
    import chardet

def fix_json_file(filename):
    """Fix encoding issues in a JSON file."""
    print(f"Processing {filename}...")
    
    # Create backup
    backup_file = f"{filename}.backup"
    shutil.copy2(filename, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Read file in binary mode
    with open(filename, "rb") as f:
        raw_data = f.read()
    
    # Detect encoding
    encoding_result = chardet.detect(raw_data)
    encoding = encoding_result["encoding"]
    confidence = encoding_result["confidence"]
    print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
    
    # Decode with detected encoding
    if encoding:
        try:
            text = raw_data.decode(encoding)
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}, trying with errors='replace'")
            text = raw_data.decode(encoding, errors='replace')
    else:
        print("Could not detect encoding, trying UTF-8 with errors='replace'")
        text = raw_data.decode('utf-8', errors='replace')
    
    # Parse as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Attempting to fix common JSON issues...")
        
        # Try to fix common JSON issues
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\\u0000', '')  # Remove escaped null bytes
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Still unable to parse JSON: {e}")
            print(f"Please check the backup file: {backup_file}")
            return False
    
    # Write back as UTF-8
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully converted {filename} to UTF-8")
    return True

def main():
    """Fix JSON files in the current directory."""
    print("UTF-8 JSON Fixer")
    print("================")
    
    # Check for specific files
    if len(sys.argv) > 1:
        files = sys.argv[1:]
        print(f"Fixing specified files: {files}")
    else:
        # Look for common JSON files
        common_files = ["docstore.json", "index_store.json", "graph_store.json"]
        files = [f for f in common_files if Path(f).exists()]
        
        if not files:
            # If no common files, look for all JSON files
            files = [f for f in os.listdir('.') if f.endswith('.json')]
        
        print(f"Found {len(files)} JSON files to fix: {files}")
    
    if not files:
        print("No JSON files found to fix.")
        return 1
    
    success_count = 0
    for filename in files:
        if fix_json_file(filename):
            success_count += 1
    
    print(f"Successfully fixed {success_count}/{len(files)} files")
    return 0 if success_count == len(files) else 1

if __name__ == "__main__":
    sys.exit(main())