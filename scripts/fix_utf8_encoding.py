#!/usr/bin/env python3
"""
Script to fix UTF-8 encoding issues in JSON files using chardet for encoding detection.

This script detects the current encoding of JSON files and rewrites them in proper UTF-8 encoding.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path

try:
    import chardet
except ImportError:
    print("chardet is required. Install it with: pip install chardet")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_and_fix_encoding(file_path, backup=True):
    """
    Detect encoding of a file and rewrite it in UTF-8.
    
    Args:
        file_path: Path to the file to fix
        backup: Whether to create a backup of the original file
    
    Returns:
        bool: True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist")
        return False
    
    try:
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Read file in binary mode to detect encoding
        with open(file_path, "rb") as f:
            raw_data = f.read()
        
        # Detect encoding
        encoding_result = chardet.detect(raw_data)
        detected_encoding = encoding_result["encoding"]
        confidence = encoding_result["confidence"]
        
        logger.info(f"{file_path.name} - Detected encoding: {detected_encoding} (confidence: {confidence:.2f})")
        
        # If already UTF-8 with high confidence, skip
        if detected_encoding and detected_encoding.lower() in ['utf-8', 'ascii'] and confidence > 0.9:
            logger.info(f"{file_path.name} is already in UTF-8 format")
            return True
        
        # Decode with detected encoding
        if detected_encoding:
            try:
                text_content = raw_data.decode(detected_encoding)
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode with {detected_encoding}, trying with errors='replace'")
                text_content = raw_data.decode(detected_encoding, errors='replace')
        else:
            logger.warning(f"Could not detect encoding for {file_path.name}, trying UTF-8 with errors='replace'")
            text_content = raw_data.decode('utf-8', errors='replace')
        
        # Parse as JSON to validate
        try:
            json_data = json.loads(text_content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path.name}: {str(e)}")
            return False
        
        # Write back as UTF-8
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully converted {file_path.name} to UTF-8")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return False


def fix_directory_encoding(directory_path, backup=True):
    """
    Fix encoding for all JSON files in a directory.
    
    Args:
        directory_path: Path to directory containing JSON files
        backup: Whether to create backups of original files
    
    Returns:
        bool: True if all files were processed successfully
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory {directory_path} does not exist or is not a directory")
        return False
    
    # Find all JSON files
    json_files = list(directory_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory_path}")
        return False
    
    logger.info(f"Found {len(json_files)} JSON files to process in {directory_path}")
    
    success_count = 0
    for json_file in json_files:
        if detect_and_fix_encoding(json_file, backup):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(json_files)} files")
    return success_count == len(json_files)


def fix_specific_files(file_paths, backup=True):
    """
    Fix encoding for specific files.
    
    Args:
        file_paths: List of file paths to fix
        backup: Whether to create backups of original files
    
    Returns:
        bool: True if all files were processed successfully
    """
    success_count = 0
    
    for file_path in file_paths:
        logger.info(f"Processing file: {file_path}")
        if detect_and_fix_encoding(file_path, backup):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(file_paths)} files")
    return success_count == len(file_paths)


def main():
    """Main function to fix UTF-8 encoding issues."""
    parser = argparse.ArgumentParser(description="Fix UTF-8 encoding issues in JSON files")
    parser.add_argument("--dir", type=str, 
                        help="Directory containing JSON files to fix")
    parser.add_argument("--files", nargs="+", 
                        help="Specific files to fix (e.g., docstore.json index_store.json)")
    parser.add_argument("--no-backup", action="store_true", 
                        help="Skip creating backups of original files")
    parser.add_argument("--survivalist", action="store_true",
                        help="Fix the survivalist graph directory specifically")
    
    args = parser.parse_args()
    
    backup = not args.no_backup
    
    if args.survivalist:
        # Fix the survivalist graph directory
        survivalist_dir = Path("./data/prebuilt_graphs/survivalist")
        if not survivalist_dir.exists():
            survivalist_dir = Path("./improved-local-assistant/data/prebuilt_graphs/survivalist")
        
        if survivalist_dir.exists():
            logger.info(f"Fixing survivalist graph directory: {survivalist_dir}")
            success = fix_directory_encoding(survivalist_dir, backup)
        else:
            logger.error("Survivalist graph directory not found")
            success = False
            
    elif args.dir:
        # Fix specific directory
        logger.info(f"Fixing directory: {args.dir}")
        success = fix_directory_encoding(args.dir, backup)
        
    elif args.files:
        # Fix specific files
        logger.info(f"Fixing files: {args.files}")
        success = fix_specific_files(args.files, backup)
        
    else:
        # Default: fix common JSON files in current directory
        common_files = ["docstore.json", "index_store.json", "graph_store.json"]
        existing_files = [f for f in common_files if Path(f).exists()]
        
        if existing_files:
            logger.info(f"Fixing common JSON files: {existing_files}")
            success = fix_specific_files(existing_files, backup)
        else:
            logger.error("No files specified and no common JSON files found in current directory")
            logger.info("Usage examples:")
            logger.info("  python fix_utf8_encoding.py --survivalist")
            logger.info("  python fix_utf8_encoding.py --dir /path/to/directory")
            logger.info("  python fix_utf8_encoding.py --files docstore.json index_store.json")
            success = False
    
    if success:
        logger.info("UTF-8 encoding fix completed successfully")
        return 0
    else:
        logger.error("UTF-8 encoding fix failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())