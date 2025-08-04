#!/usr/bin/env python3
"""
Quick setup script for the Improved Local Assistant.

This script helps users get started quickly by:
1. Checking system requirements
2. Verifying Ollama installation and models
3. Optionally downloading prebuilt knowledge graphs
4. Running initial system validation
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("🚀 Improved Local Assistant - Quick Setup")
    print("=" * 50)

def check_python_version():
    """Check Python version requirements."""
    print("\n🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    print("\n🤖 Checking Ollama installation...")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ Ollama installed: {result.stdout.strip()}")
        else:
            print("   ❌ Ollama command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ Ollama not found in PATH")
        print("   📥 Install from: https://ollama.ai")
        return False
    
    # Check if Ollama service is running
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama service is running")
            return True
        else:
            print("   ⚠️  Ollama service not responding")
            return False
    except requests.RequestException:
        print("   ⚠️  Ollama service not running (start with 'ollama serve')")
        return False

def check_required_models():
    """Check if required models are available."""
    print("\n📦 Checking required models...")
    
    required_models = ['hermes3:3b', 'phi3:mini']
    available_models = []
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
    except requests.RequestException:
        print("   ❌ Cannot check models (Ollama not running)")
        return False
    
    missing_models = []
    for model in required_models:
        if any(model in available for available in available_models):
            print(f"   ✅ {model} available")
        else:
            print(f"   ❌ {model} missing")
            missing_models.append(model)
    
    if missing_models:
        print(f"\n   📥 To install missing models:")
        for model in missing_models:
            print(f"      ollama pull {model}")
        return False
    
    return True

def check_dependencies():
    """Check if Python dependencies are installed."""
    print("\n📚 Checking Python dependencies...")
    
    try:
        import fastapi
        import llama_index
        import requests
        import psutil
        print("   ✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        print("   📥 Install with: pip install -r requirements.txt")
        return False

def offer_graph_download():
    """Offer to download prebuilt knowledge graphs."""
    print("\n🗂️  Prebuilt Knowledge Graphs")
    print("   Prebuilt graphs provide ready-to-use knowledge bases")
    print("   Available: survivalist, medical, technical")
    
    response = input("\n   Download prebuilt graphs? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n   📦 Starting graph download...")
        try:
            # Import and run the download script
            sys.path.append(str(Path(__file__).parent))
            from download_graphs import main as download_main
            
            # Set up arguments for automatic download
            original_argv = sys.argv
            sys.argv = ['download_graphs.py', 'all']
            
            download_main()
            
            sys.argv = original_argv
            print("   ✅ Graphs downloaded successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Graph download failed: {e}")
            print("   💡 You can download later with: python scripts/download_graphs.py")
            return False
    else:
        print("   ⏭️  Skipped graph download")
        print("   💡 Download later with: python scripts/download_graphs.py")
        return True

def run_system_validation():
    """Run system validation tests."""
    print("\n🔍 Running system validation...")
    
    try:
        # Run the milestone validation
        result = subprocess.run([
            sys.executable, 'cli/validate_milestone_6.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   ✅ System validation passed")
            return True
        else:
            print("   ⚠️  System validation had issues:")
            print(f"      {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  System validation timed out")
        return False
    except FileNotFoundError:
        print("   ⚠️  Validation script not found")
        return False

def print_next_steps(all_checks_passed):
    """Print next steps for the user."""
    print("\n🎯 Setup Complete!")
    print("=" * 30)
    
    if all_checks_passed:
        print("✅ All checks passed! You're ready to go.")
        print("\n🚀 Next steps:")
        print("   1. Start the application:")
        print("      python run_app.py")
        print("\n   2. Open your browser:")
        print("      http://localhost:8000")
        print("\n   3. Try the GraphRAG REPL:")
        print("      python cli/graphrag_repl.py")
    else:
        print("⚠️  Some issues were found. Please address them and run setup again.")
        print("\n🔧 Common solutions:")
        print("   • Install Ollama: https://ollama.ai")
        print("   • Pull models: ollama pull hermes3:3b && ollama pull phi3:mini")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Start Ollama service: ollama serve")

def main():
    """Main setup process."""
    print_banner()
    
    checks = [
        ("Python Version", check_python_version),
        ("Ollama Installation", check_ollama_installation),
        ("Required Models", check_required_models),
        ("Python Dependencies", check_dependencies),
    ]
    
    all_passed = True
    
    # Run all checks
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    # Offer graph download if basic checks passed
    if all_passed:
        offer_graph_download()
        
        # Run system validation
        print("\n" + "="*50)
        validation_passed = run_system_validation()
        all_passed = all_passed and validation_passed
    
    # Print final results
    print_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())