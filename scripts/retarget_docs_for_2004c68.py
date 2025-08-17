#!/usr/bin/env python3
"""
Script to retarget documentation for the 2004c68 commit layout.

This script updates README and documentation files to use commands and paths
that match the August 13, 2025 baseline (commit 2004c68) structure.
"""

import pathlib
import re
import sys

def main():
    """Main function to retarget docs."""
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    
    if not readme.exists():
        print(f"README.md not found at {readme}")
        return 1
    
    print(f"Processing {readme}")
    text = readme.read_text(encoding="utf-8")
    
    # Define replacement rules
    rules = [
        # Replace ila CLI commands with direct uvicorn/python commands
        (r"\bila api\b[^\n]*", "uvicorn app.main:app --reload"),
        (r"\bila\s+api\s+--reload", "uvicorn app.main:app --reload"),
        (r"\bila\s+api", "uvicorn app.main:app"),
        
        # Replace pip install commands
        (r"pip install -e \. -c constraints\.txt", "pip install -r requirements.txt"),
        (r"pip install -e \.", "pip install -r requirements.txt"),
        
        # Replace src/ paths with app/ paths
        (r"src/improved_local_assistant", "app"),
        (r"src/", ""),
        
        # Update any references to the new structure back to old structure
        (r"improved_local_assistant\.cli", "cli"),
        (r"from improved_local_assistant\.", "from "),
        
        # Fix any remaining path references
        (r"app/app/", "app/"),
    ]
    
    # Apply replacements
    original_text = text
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    
    # Write back if changes were made
    if text != original_text:
        readme.write_text(text, encoding="utf-8")
        print("README retargeted for 2004c68")
        print("Changes made:")
        
        # Show what changed
        for pattern, replacement in rules:
            if re.search(pattern, original_text):
                print(f"  - {pattern} -> {replacement}")
    else:
        print("No changes needed")
    
    # Also process any docs files if they exist
    docs_dir = root / "docs"
    if docs_dir.exists():
        print(f"\nProcessing docs in {docs_dir}")
        for doc_file in docs_dir.rglob("*.md"):
            if doc_file.name.lower() in ["readme.md", "quickstart.md", "installation.md"]:
                print(f"Processing {doc_file}")
                doc_text = doc_file.read_text(encoding="utf-8")
                original_doc_text = doc_text
                
                for pattern, replacement in rules:
                    doc_text = re.sub(pattern, replacement, doc_text)
                
                if doc_text != original_doc_text:
                    doc_file.write_text(doc_text, encoding="utf-8")
                    print(f"  Updated {doc_file.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())