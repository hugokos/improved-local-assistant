#!/usr/bin/env python3
"""
Final cleanup script to fix remaining lint issues.
"""

import re
import os
from pathlib import Path

def fix_file_issues(file_path):
    """Fix common issues in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Fix blank lines with whitespace
    content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Ensure file ends with newline
    if content and not content.endswith('\n'):
        content += '\n'
    
    # Fix multiple blank lines (max 2)
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed whitespace issues in {file_path}")
        return True
    return False

def remove_unused_imports(file_path):
    """Remove specific unused imports."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Patterns for unused imports to remove
    unused_patterns = [
        r'^from typing import.*Optional.*\n$',
        r'^from typing import.*List.*\n$', 
        r'^from typing import.*Tuple.*\n$',
        r'^from typing import.*Union.*\n$',
        r'^import sys\n$',
        r'^import asyncio\n$',
        r'^import time\n$',
        r'^from datetime import datetime\n$',
        r'^from dataclasses import dataclass\n$',
    ]
    
    # Only remove if they're clearly unused (this is a simplified approach)
    # In practice, you'd want to use a proper tool like autoflake
    
    original_lines = lines[:]
    
    # Remove some obvious unused imports based on the lint output
    if 'config_validator.py' in str(file_path):
        lines = [line for line in lines if not line.strip().startswith('import sys')]
    
    if lines != original_lines:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Removed unused imports from {file_path}")
        return True
    return False

def fix_long_lines(file_path):
    """Fix some long lines by breaking them."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix some common long line patterns
    # This is a simplified approach - in practice you'd use black or similar
    
    # Break long import lines
    def fix_import_line(match):
        imports = match.group(1).replace(", ", ",\n    ")
        return f'from fastapi import (\n    {imports}\n)'
    
    content = re.sub(r'from fastapi import (.{80,})', fix_import_line, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed long lines in {file_path}")
        return True
    return False

def main():
    """Run final cleanup on all Python files."""
    print("🧹 Running final cleanup...")
    
    # Get all Python files
    python_files = []
    for pattern in ['app/*.py', 'services/*.py']:
        python_files.extend(Path('.').glob(pattern))
    
    fixed_files = 0
    
    for file_path in python_files:
        print(f"\n🔧 Processing {file_path}...")
        
        changes = 0
        if fix_file_issues(file_path):
            changes += 1
        if remove_unused_imports(file_path):
            changes += 1
        if fix_long_lines(file_path):
            changes += 1
            
        if changes > 0:
            fixed_files += 1
    
    print(f"\n📊 Fixed issues in {fixed_files} files")
    
    # Run a final lint check
    print("\n🔍 Running final lint check...")
    os.system("python -m flake8 app/ services/ --max-line-length=120 --count")

if __name__ == "__main__":
    main()