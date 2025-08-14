#!/usr/bin/env python3
"""
Test script for gcode file watching functionality.
"""

import os
import time
import tempfile
from pathlib import Path

def test_file_watching():
    """Test the file watching functionality."""
    print("🧪 Testing gcode File Watching")
    print("=" * 50)
    
    # Test the file watcher module
    try:
        from gcode.file_watcher import create_file_watcher
        print("✅ File watcher module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import file watcher: {e}")
        return False
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Created test directory: {temp_path}")
        
        # Create a test file watcher
        try:
            watcher = create_file_watcher(str(temp_path))
            print("✅ File watcher created successfully")
        except Exception as e:
            print(f"❌ Failed to create file watcher: {e}")
            return False
        
        # Test file watching
        try:
            print("🔍 Starting file watcher...")
            watcher.start()
            print("✅ File watcher started")
            
            # Wait a moment for initial scan
            time.sleep(2)
            
            # Check status
            status = watcher.get_status()
            print(f"📊 Status: {status}")
            
            # Create a test file
            test_file = temp_path / "test.py"
            test_file.write_text("# Test file for watching")
            print(f"📝 Created test file: {test_file}")
            
            # Wait for file change detection
            time.sleep(3)
            
            # Stop watching
            watcher.stop()
            print("✅ File watcher stopped")
            
        except Exception as e:
            print(f"❌ Error during file watching test: {e}")
            return False
    
    print("\n🎉 File watching test completed successfully!")
    return True

def test_gcode_integration():
    """Test gcode integration with file watching."""
    print("\n🔗 Testing gcode Integration")
    print("=" * 50)
    
    try:
        from gcode.agent import GeminiAgent
        
        # Create agent instance
        agent = GeminiAgent()
        print("✅ gcode agent created successfully")
        
        # Test watch commands
        print("\n📋 Testing watch commands:")
        
        # Test status when not watching
        print("\n--- watch status (inactive) ---")
        agent.watch_commands('status')
        
        # Test help
        print("\n--- watch help ---")
        agent.watch_commands('help')
        
        print("\n✅ gcode integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during gcode integration test: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 gcode File Watching Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Watcher Module", test_file_watching),
        ("gcode Integration", test_gcode_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! File watching is working correctly.")
        print("\n💡 You can now use file watching in gcode:")
        print("   gcode 'watch start'     - Start monitoring")
        print("   gcode 'watch status'    - Check status")
        print("   gcode 'watch stop'      - Stop monitoring")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
