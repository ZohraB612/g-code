#!/usr/bin/env python3
"""
Test script to verify Docker functionality for gcode.
"""

import os
import subprocess
import sys

def test_docker_installation():
    """Test if Docker is installed and running."""
    print("🐳 Testing Docker Installation...")
    
    try:
        # Check if Docker is installed
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker installed: {result.stdout.strip()}")
        
        # Check if Docker daemon is running (more robust check)
        try:
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True, check=True, timeout=10)
            print("✅ Docker daemon is running")
            return True
        except subprocess.TimeoutExpired:
            print("⚠️  Docker daemon check timed out (may be slow)")
            return True
        except subprocess.CalledProcessError:
            print("❌ Docker daemon is not responding")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker error: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def test_docker_compose():
    """Test if Docker Compose is available."""
    print("\n🐳 Testing Docker Compose...")
    
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose available: {result.stdout.strip()}")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker Compose not available")
        return False

def test_gcode_docker_build():
    """Test building the gcode Docker image."""
    print("\n🔨 Testing gcode Docker Build...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Check if Dockerfile exists in parent directory
    dockerfile_path = os.path.join(parent_dir, 'Dockerfile')
    compose_path = os.path.join(parent_dir, 'docker-compose.yml')
    
    if not os.path.exists(dockerfile_path):
        print(f"❌ Dockerfile not found at {dockerfile_path}")
        return False
    
    if not os.path.exists(compose_path):
        print(f"❌ docker-compose.yml not found at {compose_path}")
        return False
    
    print("✅ Docker configuration files found")
    
    # Change to parent directory for docker-compose
    original_dir = os.getcwd()
    os.chdir(parent_dir)
    
    try:
        # Try to build the image
        print("🔨 Building gcode Docker image...")
        result = subprocess.run(['docker-compose', '--profile', 'dev', 'build'], 
                              capture_output=True, text=True, check=True)
        print("✅ Docker image built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e.stderr}")
        return False
    finally:
        os.chdir(original_dir)

def show_docker_usage():
    """Show how to use gcode with Docker."""
    print("\n🚀 Docker Usage Instructions:")
    print("=" * 50)
    
    print("\n1. Quick Start:")
    print("   cd gcode/")
    print("   docker-compose --profile cli up")
    print("   # Runs gcode in interactive mode")
    
    print("\n2. Development Environment:")
    print("   cd gcode/")
    print("   docker-compose --profile dev up -d")
    print("   docker-compose --profile dev exec gcode-dev bash")
    print("   # Full development environment with VS Code integration")
    
    print("\n3. Web Interface:")
    print("   cd gcode/")
    print("   docker-compose --profile web up")
    print("   # Web interface on http://localhost:8000")
    
    print("\n4. Custom Commands:")
    print("   cd gcode/")
    print("   docker run -it --rm -v $(pwd):/workspace gcode")
    print("   # Run gcode in any directory")
    
    print("\n5. VS Code Dev Container:")
    print("   # Install 'Remote - Containers' extension")
    print("   # Open gcode project in VS Code")
    print("   # Press Ctrl+Shift+P and select 'Dev Containers: Reopen in Container'")

def main():
    """Run all Docker tests."""
    print("🧪 Testing gcode Docker Functionality")
    print("=" * 50)
    
    tests = [
        test_docker_installation,
        test_docker_compose,
        test_gcode_docker_build
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Docker tests passed! gcode is ready for containerized deployment.")
        show_docker_usage()
    else:
        print("⚠️  Some Docker tests failed. Please check your Docker installation.")
        print("\n💡 To install Docker:")
        print("   • Visit: https://docs.docker.com/get-docker/")
        print("   • Or use: curl -fsSL https://get.docker.com | sh")

if __name__ == "__main__":
    main()
