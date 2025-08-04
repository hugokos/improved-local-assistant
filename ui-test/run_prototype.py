#!/usr/bin/env python3
"""
Quick launcher for Streamlit prototypes
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch a Streamlit prototype"""
    
    # Available prototypes
    prototypes = {
        "1": ("chat_interface.py", "Chat Interface"),
        "2": ("dashboard.py", "Dashboard"),
        "3": ("model_manager.py", "Model Manager")
    }
    
    print("🚀 Streamlit Prototype Launcher")
    print("=" * 40)
    
    for key, (file, name) in prototypes.items():
        print(f"{key}. {name}")
    
    print("\nEnter prototype number (or 'q' to quit): ", end="")
    choice = input().strip()
    
    if choice.lower() == 'q':
        print("Goodbye!")
        return
    
    if choice not in prototypes:
        print("Invalid choice!")
        return
    
    prototype_file, prototype_name = prototypes[choice]
    prototype_path = Path(__file__).parent / "streamlit_prototypes" / prototype_file
    
    if not prototype_path.exists():
        print(f"Error: {prototype_file} not found!")
        return
    
    print(f"\n🎯 Launching {prototype_name}...")
    print(f"📁 File: {prototype_path}")
    print("🌐 Opening in browser...")
    print("\n" + "=" * 40)
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Change to the prototype directory
        os.chdir(prototype_path.parent)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            prototype_file, 
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Prototype stopped!")
    except Exception as e:
        print(f"\n❌ Error launching prototype: {e}")

if __name__ == "__main__":
    main()