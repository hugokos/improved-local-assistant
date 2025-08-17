#!/usr/bin/env python3
"""
Post-installation script for Improved Local Assistant.

This script runs after pip install and ensures all required models are downloaded.
"""

import subprocess
import sys
from pathlib import Path


def download_voice_models():
    """Download voice models if they don't exist."""
    print("🎵 Checking voice models...")
    
    # Check if voice models exist
    models_dir = Path("models")
    vosk_model = models_dir / "vosk" / "vosk-model-small-en"
    piper_model = models_dir / "piper" / "en_US-lessac-medium.onnx"
    
    if vosk_model.exists() and piper_model.exists():
        print("   ✅ Voice models already exist")
        return True
    
    print("   📥 Downloading voice models...")
    
    try:
        # Run the download script
        script_path = Path(__file__).parent / "download_voice_models.py"
        
        if not script_path.exists():
            print("   ⚠️  Voice model download script not found")
            return False
        
        # Download both vosk and piper models
        result = subprocess.run([
            sys.executable, str(script_path), 
            "--vosk", "small-en",
            "--piper", "en_US-lessac-medium"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✅ Voice models downloaded successfully")
            return True
        else:
            print(f"   ⚠️  Voice model download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Voice model download timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error downloading voice models: {e}")
        return False


def main():
    """Main post-install process."""
    print("🚀 Running post-installation setup...")
    
    success = True
    
    # Download voice models
    if not download_voice_models():
        success = False
        print("   💡 You can download voice models later with:")
        print("      python scripts/download_voice_models.py --all")
    
    if success:
        print("✅ Post-installation setup completed successfully!")
    else:
        print("⚠️  Post-installation setup completed with warnings")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())