#!/usr/bin/env python3
"""
Fix import issues by converting Python 3.10+ union syntax to Python 3.9 compatible syntax.
"""

import pathlib
import re
from typing import List, Tuple, Optional, Union

ROOT = pathlib.Path(__file__).resolve().parents[1]

def want(p: pathlib.Path) -> bool:
    """Check if we should process this Python file."""
    parts = set(p.parts)
    return (
        p.suffix == ".py"
        and ".venv" not in parts and "venv" not in parts
        and "site-packages" not in parts and "dist" not in parts
        and "__pycache__" not in parts
    )

def fix_union_syntax(text: str) -> Tuple[str, List[str]]:
    """Fix Python 3.10+ union syntax to be compatible with Python 3.9."""
    changes = []
    new_text = text
    
    # Pattern to match type annotations with | union syntax
    # This matches patterns like: var: Optional[str], param: Union[int, float], etc.
    union_pattern = r':\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*\|\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*(?:\s*\|\s*[A-Za-z_][A-Za-z0-9_\.\[\]]*)*)'
    
    def replace_union(match):
        full_match = match.group(0)
        types = full_match.split(':')[1].strip()
        
        # Split by | and clean up
        type_parts = [t.strip() for t in types.split('|')]
        
        # Handle common cases
        if len(type_parts) == 2 and 'None' in type_parts:
            # This is Optional[T]
            non_none_type = [t for t in type_parts if t != 'None'][0]
            replacement = f': Optional[{non_none_type}]'
            changes.append(f"  {full_match} -> {replacement}")
            return replacement
        else:
            # This is Union[T1, T2, ...]
            union_types = ', '.join(type_parts)
            replacement = f': Union[{union_types}]'
            changes.append(f"  {full_match} -> {replacement}")
            return replacement
    
    new_text = re.sub(union_pattern, replace_union, new_text)
    
    # Also fix function return type annotations
    return_union_pattern = r'->\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*\|\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*(?:\s*\|\s*[A-Za-z_][A-Za-z0-9_\.\[\]]*)*)'
    
    def replace_return_union(match):
        full_match = match.group(0)
        types = full_match.split('->')[1].strip()
        
        # Split by | and clean up
        type_parts = [t.strip() for t in types.split('|')]
        
        # Handle common cases
        if len(type_parts) == 2 and 'None' in type_parts:
            # This is Optional[T]
            non_none_type = [t for t in type_parts if t != 'None'][0]
            replacement = f'-> Optional[{non_none_type}]'
            changes.append(f"  {full_match} -> {replacement}")
            return replacement
        else:
            # This is Union[T1, T2, ...]
            union_types = ', '.join(type_parts)
            replacement = f'-> Union[{union_types}]'
            changes.append(f"  {full_match} -> {replacement}")
            return replacement
    
    new_text = re.sub(return_union_pattern, replace_return_union, new_text)
    
    return new_text, changes

def add_typing_imports(text: str, changes: List[str]) -> str:
    """Add necessary typing imports if changes were made."""
    if not changes:
        return text
    
    needs_optional = any('Optional[' in change for change in changes)
    needs_union = any('Union[' in change for change in changes)
    
    if not (needs_optional or needs_union):
        return text
    
    # Check if typing import already exists
    has_typing_import = re.search(r'^from typing import', text, re.MULTILINE)
    has_typing_import_as = re.search(r'^import typing', text, re.MULTILINE)
    
    imports_to_add = []
    if needs_optional:
        imports_to_add.append('Optional')
    if needs_union:
        imports_to_add.append('Union')
    
    if has_typing_import:
        # Add to existing from typing import
        import_line_match = re.search(r'^from typing import (.+)$', text, re.MULTILINE)
        if import_line_match:
            existing_imports = import_line_match.group(1)
            # Check what's already imported
            existing_list = [imp.strip() for imp in existing_imports.split(',')]
            
            for imp in imports_to_add:
                if imp not in existing_list:
                    existing_list.append(imp)
            
            new_import_line = f"from typing import {', '.join(existing_list)}"
            text = text.replace(import_line_match.group(0), new_import_line)
    elif has_typing_import_as:
        # Already has import typing, no need to add
        pass
    else:
        # Add new typing import at the top
        lines = text.split('\n')
        
        # Find the right place to insert (after docstring, before other imports)
        insert_idx = 0
        in_docstring = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                elif stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                    insert_idx = i + 1
            elif not in_docstring and (stripped.startswith('import ') or stripped.startswith('from ')):
                insert_idx = i
                break
            elif not in_docstring and stripped and not stripped.startswith('#'):
                insert_idx = i
                break
        
        import_line = f"from typing import {', '.join(imports_to_add)}"
        lines.insert(insert_idx, import_line)
        text = '\n'.join(lines)
    
    return text

def main():
    """Fix all Python files with union syntax issues."""
    print("🔧 Fixing Python 3.10+ union syntax for Python 3.9 compatibility")
    print("=" * 60)
    
    total_files = 0
    fixed_files = 0
    
    for py_file in filter(want, ROOT.rglob("*.py")):
        total_files += 1
        
        try:
            text = py_file.read_text(encoding="utf-8")
            new_text, changes = fix_union_syntax(text)
            
            if changes:
                # Add necessary typing imports
                new_text = add_typing_imports(new_text, changes)
                
                py_file.write_text(new_text, encoding="utf-8")
                fixed_files += 1
                
                print(f"✅ Fixed {py_file.relative_to(ROOT)}")
                for change in changes:
                    print(change)
                print()
        
        except Exception as e:
            print(f"❌ Error processing {py_file.relative_to(ROOT)}: {e}")
    
    print("=" * 60)
    print(f"📊 Processed {total_files} Python files")
    print(f"🔧 Fixed {fixed_files} files with union syntax issues")
    
    if fixed_files > 0:
        print("\n🎉 All union syntax issues should now be fixed!")
        print("Run the import smoke test again to verify: python scripts/import_smoke_test.py")
    else:
        print("\n✅ No union syntax issues found!")

if __name__ == "__main__":
    main()