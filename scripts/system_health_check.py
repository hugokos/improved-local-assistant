#!/usr/bin/env python3
"""
System Health Check Script for Improved Local Assistant

This script checks for common issues and provides recommendations for fixes.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_ollama_status():
    """Check if Ollama is running and accessible."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"✅ Ollama is running with {len(models)} models available")
            
            # Check for specific models
            model_names = [model.get("name", "") for model in models]
            required_models = ["hermes3:3b", "tinyllama"]
            
            for model in required_models:
                if any(model in name for name in model_names):
                    logger.info(f"✅ Model {model} is available")
                else:
                    logger.warning(f"⚠️  Model {model} not found. Consider running: ollama pull {model}")
            
            return True
        else:
            logger.error(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ollama is not accessible: {e}")
        logger.info("💡 Try starting Ollama with: ollama serve")
        return False


def check_gpu_memory():
    """Check GPU memory availability."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                free, total = map(int, line.split(', '))
                usage_percent = ((total - free) / total) * 100
                
                logger.info(f"🖥️  GPU {i}: {free}MB free / {total}MB total ({usage_percent:.1f}% used)")
                
                if free < 2000:  # Less than 2GB free
                    logger.warning(f"⚠️  GPU {i} has low memory ({free}MB free). This may cause CUDA out of memory errors.")
                    logger.info("💡 Consider using smaller models or CPU-only mode")
                
            return True
        else:
            logger.info("ℹ️  NVIDIA GPU not detected or nvidia-smi not available")
            return False
            
    except FileNotFoundError:
        logger.info("ℹ️  nvidia-smi not found. GPU monitoring not available.")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error checking GPU memory: {e}")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        "llama-index-readers-file",
        "sentence-transformers",
        "networkx",
        "pyvis",
        "fastapi",
        "uvicorn",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is missing")
    
    if missing_packages:
        logger.info(f"💡 Install missing packages with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_data_directories():
    """Check if required data directories exist."""
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "data/prebuilt_graphs",
        "data/dynamic_graph",
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if full_path.exists():
            logger.info(f"✅ Directory {dir_path} exists")
            
            # Check if it has content
            if any(full_path.iterdir()):
                logger.info(f"ℹ️  Directory {dir_path} contains files")
            else:
                logger.warning(f"⚠️  Directory {dir_path} is empty")
        else:
            logger.error(f"❌ Directory {dir_path} does not exist")
            logger.info(f"💡 Create with: mkdir -p {full_path}")
            all_exist = False
    
    return all_exist


def check_encoding_issues():
    """Check for potential encoding issues in graph files."""
    base_dir = Path(__file__).parent.parent
    graph_dir = base_dir / "data" / "prebuilt_graphs"
    
    if not graph_dir.exists():
        logger.info("ℹ️  No prebuilt graphs directory to check for encoding issues")
        return True
    
    encoding_issues = []
    
    for json_file in graph_dir.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                f.read()
            logger.debug(f"✅ {json_file.relative_to(base_dir)} has valid UTF-8 encoding")
        except UnicodeDecodeError as e:
            encoding_issues.append(str(json_file.relative_to(base_dir)))
            logger.warning(f"⚠️  {json_file.relative_to(base_dir)} has encoding issues: {e}")
    
    if encoding_issues:
        logger.info("💡 Encoding issues detected. The system has fallback mechanisms to handle these.")
        logger.info("💡 Consider running the encoding fix script if problems persist.")
        return False
    else:
        logger.info("✅ No encoding issues detected in graph files")
        return True


def check_system_resources():
    """Check system resource availability."""
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_percent = memory.percent
        
        logger.info(f"💾 System Memory: {memory_available_gb:.1f}GB available / {memory_gb:.1f}GB total ({memory_percent:.1f}% used)")
        
        if memory_available_gb < 4:
            logger.warning("⚠️  Low system memory available. This may affect performance.")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        logger.info(f"💿 Disk Space: {disk_free_gb:.1f}GB free / {disk_total_gb:.1f}GB total ({disk_percent:.1f}% used)")
        
        if disk_free_gb < 5:
            logger.warning("⚠️  Low disk space available. This may affect graph storage.")
        
        return True
        
    except ImportError:
        logger.warning("⚠️  psutil not available. Cannot check system resources.")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error checking system resources: {e}")
        return False


def main():
    """Run all health checks."""
    logger.info("🔍 Starting System Health Check for Improved Local Assistant")
    logger.info("=" * 60)
    
    checks = [
        ("Ollama Status", check_ollama_status),
        ("GPU Memory", check_gpu_memory),
        ("Required Packages", check_required_packages),
        ("Data Directories", check_data_directories),
        ("Encoding Issues", check_encoding_issues),
        ("System Resources", check_system_resources),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n🔍 Checking {check_name}...")
        try:
            results[check_name] = check_func()
        except Exception as e:
            logger.error(f"❌ Error during {check_name} check: {e}")
            results[check_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 HEALTH CHECK SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {check_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All checks passed! Your system is healthy.")
    else:
        logger.info("⚠️  Some issues detected. Review the recommendations above.")
        logger.info("💡 Most issues have fallback mechanisms and won't prevent basic functionality.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)