"""
Runtime UTF-8 patch for Windows compatibility.
Apply this patch at startup to ensure all file operations default to UTF-8.
"""

import builtins
import io


def apply_utf8_patch():
    """
    Monkey-patch builtins.open to default to UTF-8 encoding.
    This fixes Windows cp1252 issues when loading knowledge graphs.
    """
    # Store original open function
    _original_open = builtins.open
    
    def utf8_open(file, mode="r", *args, **kwargs):
        # Default to UTF-8 for text mode, leave binary mode alone
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _original_open(file, mode, *args, **kwargs)
    
    # Replace builtins.open globally
    builtins.open = utf8_open
    
    print("✅ UTF-8 runtime patch applied - all file operations will use UTF-8 encoding")


def verify_utf8_patch():
    """Verify the UTF-8 patch is working correctly"""
    import tempfile
    import json
    
    # Test with Unicode content
    test_data = {"message": "Hello 世界 café naïve résumé"}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, ensure_ascii=False)
        temp_path = f.name
    
    try:
        # Try to read back without explicit encoding
        with open(temp_path, 'r') as f:  # Should use UTF-8 due to patch
            loaded_data = json.load(f)
        
        if loaded_data == test_data:
            print("✅ UTF-8 patch verification passed")
            return True
        else:
            print("❌ UTF-8 patch verification failed - data mismatch")
            return False
            
    except UnicodeDecodeError:
        print("❌ UTF-8 patch verification failed - encoding error")
        return False
    finally:
        import os
        try:
            os.unlink(temp_path)
        except:
            pass


if __name__ == "__main__":
    print("🔧 Testing UTF-8 Runtime Patch")
    apply_utf8_patch()
    verify_utf8_patch()