#!/bin/bash

# gcode Development Container Runner
# Similar to Claude Code's devcontainer setup

set -e

echo "🚀 Starting gcode Development Container..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Build and start the development container
echo "🔨 Building gcode development container..."
docker-compose --profile dev build

echo "🚀 Starting gcode development environment..."
docker-compose --profile dev up -d

# Get container ID
CONTAINER_ID=$(docker-compose --profile dev ps -q gcode-dev)

if [ -z "$CONTAINER_ID" ]; then
    echo "❌ Failed to start container. Check docker-compose logs."
    exit 1
fi

echo "✅ gcode development container is running!"
echo "📦 Container ID: $CONTAINER_ID"
echo ""
echo "🔧 Available commands:"
echo "  • docker-compose --profile dev exec gcode-dev bash    - Enter container shell"
echo "  • docker-compose --profile dev exec gcode-dev gcode   - Run gcode in container"
echo "  • docker-compose --profile dev logs gcode-dev         - View container logs"
echo "  • docker-compose --profile dev down                   - Stop container"
echo ""
echo "💡 To start using gcode in the container:"
echo "   docker-compose --profile dev exec gcode-dev gcode"
echo ""
echo "🎯 The container has access to:"
echo "   • Your current project directory (mounted at /app)"
echo "   • Git configuration"
echo "   • SSH keys (if available)"
echo "   • gcode configuration in .gcode/"

# Optionally enter the container
read -p "🤔 Would you like to enter the container now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔌 Entering gcode development container..."
    docker-compose --profile dev exec gcode-dev bash
fi
