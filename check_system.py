#!/usr/bin/env python3
"""
System requirements checker for SkyReels-V2
"""

import sys
import subprocess
import importlib
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🔥 GPU Check:")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'MiB' in line and 'GPU' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no NVIDIA GPU or drivers")
        return False

def check_memory():
    """Check system memory"""
    print("\n💾 Memory Check:")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"   Total RAM: {total_gb:.1f} GB")
        print(f"   Available: {available_gb:.1f} GB")
        
        if total_gb >= 16:
            print("✅ Sufficient RAM for SkyReels-V2")
            return True
        else:
            print("⚠️  Less than 16GB RAM - may have issues with large models")
            return False
            
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
        return True

def check_disk_space():
    """Check available disk space"""
    print("\n💽 Disk Space Check:")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        
        print(f"   Free space: {free_gb:.1f} GB")
        
        if free_gb >= 50:
            print("✅ Sufficient disk space")
            return True
        else:
            print("⚠️  Less than 50GB free - may need more space for models")
            return False
            
    except Exception as e:
        print(f"⚠️  Cannot check disk space: {e}")
        return True

def check_environment():
    """Check what environment we're running in"""
    print("\n🌍 Environment Check:")
    
    if 'COLAB_GPU' in os.environ:
        print("✅ Google Colab detected")
        return "colab"
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("✅ Kaggle detected")
        return "kaggle"
    elif 'JPY_PARENT_PID' in os.environ:
        print("✅ Jupyter environment detected")
        return "jupyter"
    else:
        print("✅ Standard Python environment")
        return "standard"

def check_internet():
    """Check internet connectivity"""
    print("\n🌐 Internet Check:")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print("✅ Internet connection available")
        print("✅ Can access Hugging Face (needed for model downloads)")
        return True
    except Exception as e:
        print(f"❌ Internet connection issue: {e}")
        return False

def main():
    """Run all checks"""
    print("🔍 SkyReels-V2 System Requirements Check")
    print("=" * 45)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu),
        ("System Memory", check_memory),
        ("Disk Space", check_disk_space),
        ("Environment", lambda: check_environment() is not None),
        ("Internet Connection", check_internet),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 45)
    print("📊 Summary:")
    
    passed = 0
    for name, result in results:
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            if result:
                passed += 1
        else:
            status = "✅ DETECTED"
            passed += 1
        print(f"   {status} {name}")
    
    print(f"\nPassed: {passed}/{len(results)} checks")
    
    if passed >= len(results) - 1:  # Allow one failure
        print("\n🎉 System looks ready for SkyReels-V2!")
        print("\nRecommendations:")
        print("- Use 1.3B models if you have 16GB GPU VRAM")
        print("- Use 14B models if you have 24GB+ GPU VRAM")
        print("- Enable TeaCache for faster generation")
        print("- Start with short videos (4-10 seconds)")
    else:
        print("\n⚠️  Some issues detected. SkyReels-V2 may still work but with limitations.")
        print("\nSuggestions:")
        print("- Install NVIDIA drivers if GPU check failed")
        print("- Free up disk space if needed")
        print("- Consider using smaller models or CPU-only mode")
    
    return passed >= len(results) - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
