#!/usr/bin/env python3
"""
Quick fix script for common startup issues.

This script addresses the most common startup problems:
1. Wrong embedding model ID
2. Ollama model listing issues
3. Windows process priority warnings
4. Missing dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_and_fix_embedding_model():
    """Check and fix embedding model configuration."""
    print("🔍 Checking embedding model configuration...")
    
    # Check if the correct model is configured
    config_file = Path("config.yaml")
    if config_file.exists():
        content = config_file.read_text()
        if "sentence-transformers/bge-small-en-v1.5" in content:
            print("❌ Found incorrect embedding model ID")
            content = content.replace(
                "sentence-transformers/bge-small-en-v1.5", 
                "BAAI/bge-small-en-v1.5"
            )
            config_file.write_text(content)
            print("✅ Fixed embedding model ID in config.yaml")
        elif "BAAI/bge-small-en-v1.5" in content:
            print("✅ Embedding model ID is correct")
        else:
            print("⚠️  No embedding model configuration found")
    else:
        print("⚠️  config.yaml not found")

def check_ollama_connection():
    """Check Ollama connection and models."""
    print("\n🔍 Checking Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = []
            for m in models.get('models', []):
                model_name = m.get('model') or m.get('name', '')
                if model_name:
                    available_models.append(model_name)
            
            print(f"✅ Ollama is running with {len(available_models)} models")
            print(f"📋 Available models: {', '.join(available_models)}")
            
            # Check required models (handle :latest suffix)
            required_models = ["hermes3:3b", "tinyllama"]
            available_base_models = [m.split(':')[0] for m in available_models]
            missing_models = []
            
            for required in required_models:
                base_required = required.split(':')[0]
                if base_required not in available_base_models and required not in available_models:
                    missing_models.append(required)
            
            if missing_models:
                print(f"❌ Missing required models: {', '.join(missing_models)}")
                print("💡 Run: ollama pull hermes3:3b && ollama pull tinyllama")
            else:
                print("✅ All required models are available")
                
        else:
            print(f"❌ Ollama API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama (is it running?)")
        print("💡 Start Ollama with: ollama serve")
    except ImportError:
        print("❌ requests library not available")
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")

def check_dependencies():
    """Check and install missing dependencies."""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        "sentence-transformers",
        "torch",
        "transformers",
        "ollama",
        "fastapi",
        "uvicorn",
        "websockets"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n💡 Install missing packages: pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required packages are installed")

def fix_windows_issues():
    """Fix Windows-specific issues."""
    if platform.system() != "Windows":
        return
        
    print("\n🔍 Checking Windows-specific issues...")
    
    # Check for win32api
    try:
        import win32api
        print("✅ win32api is available")
    except ImportError:
        print("⚠️  win32api not available (process priority adjustment disabled)")
        print("💡 Install with: pip install pywin32")
    
    # Set environment variables for Windows
    os.environ["SKIP_PROCESS_PRIORITY"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    print("✅ Set Windows environment variables")

def download_embedding_model():
    """Download embedding model for offline use."""
    print("\n🔍 Checking embedding model availability...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"📥 Downloading {model_name}...")
        
        # This will download the model if not already cached
        model = SentenceTransformer(model_name, device="cpu")
        print("✅ Embedding model is available")
        
        # Test the model
        test_embedding = model.encode("test sentence")
        print(f"✅ Model test successful (embedding size: {len(test_embedding)})")
        
    except Exception as e:
        print(f"❌ Error with embedding model: {e}")
        print("💡 Try: pip install sentence-transformers torch")

def create_directories():
    """Create necessary directories."""
    print("\n🔍 Creating necessary directories...")
    
    directories = [
        "logs",
        "data",
        "data/prebuilt_graphs",
        "data/dynamic_graph", 
        "data/sessions"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def main():
    """Run all fixes."""
    print("🚀 Improved Local AI Assistant - Startup Issue Fixer")
    print("=" * 50)
    
    # Change to project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Run all checks and fixes
    check_and_fix_embedding_model()
    check_ollama_connection()
    check_dependencies()
    fix_windows_issues()
    download_embedding_model()
    create_directories()
    
    print("\n" + "=" * 50)
    print("🎉 Startup issue fixes completed!")
    print("\n💡 Now try running: python app.py --preload models --lazy-load-graphs")

if __name__ == "__main__":
    main()