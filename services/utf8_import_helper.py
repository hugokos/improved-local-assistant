"""
UTF-8 Import Helper - Ensures consistent graph_builder import path setup
"""
import os
import sys
from pathlib import Path

def setup_graph_builder_path():
    """
    Set up the Python path to import graph_builder module from kg_builder/src
    """
    # Get the path to kg_builder/src relative to this file
    current_dir = Path(__file__).parent
    kg_builder_path = current_dir.parent.parent / "kg_builder" / "src"
    
    # Add to Python path if not already there
    kg_builder_str = str(kg_builder_path.resolve())
    if kg_builder_str not in sys.path:
        sys.path.insert(0, kg_builder_str)

def get_utf8_filesystem():
    """
    Import and return UTF8FileSystem with proper path setup
    """
    setup_graph_builder_path()
    try:
        from graph_builder import UTF8FileSystem
        return UTF8FileSystem()
    except ImportError as e:
        # Fallback - create a simple UTF-8 filesystem
        from fsspec.implementations.local import LocalFileSystem
        
        class UTF8FileSystem(LocalFileSystem):
            """LocalFileSystem that defaults to UTF‑8 for text writes."""
            def open(self, path: str, mode: str = "rb", **kwargs):
                if "b" not in mode:
                    kwargs.setdefault("encoding", "utf-8")
                return super().open(path, mode=mode, **kwargs)
        
        return UTF8FileSystem()