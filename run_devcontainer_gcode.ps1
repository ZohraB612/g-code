# gcode Development Container Runner (PowerShell)
# Similar to Claude Code's devcontainer setup

param(
    [switch]$EnterContainer
)

Write-Host "🚀 Starting gcode Development Container..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker and try again." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ docker-compose is not installed. Please install it and try again." -ForegroundColor Red
    exit 1
}

# Build and start the development container
Write-Host "🔨 Building gcode development container..." -ForegroundColor Yellow
docker-compose --profile dev build

Write-Host "🚀 Starting gcode development environment..." -ForegroundColor Yellow
docker-compose --profile dev up -d

# Get container ID
$CONTAINER_ID = docker-compose --profile dev ps -q gcode-dev

if (-not $CONTAINER_ID) {
    Write-Host "❌ Failed to start container. Check docker-compose logs." -ForegroundColor Red
    exit 1
}

Write-Host "✅ gcode development container is running!" -ForegroundColor Green
Write-Host "📦 Container ID: $CONTAINER_ID" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔧 Available commands:" -ForegroundColor Yellow
Write-Host "  • docker-compose --profile dev exec gcode-dev bash    - Enter container shell" -ForegroundColor White
Write-Host "  • docker-compose --profile dev exec gcode-dev gcode   - Run gcode in container" -ForegroundColor White
Write-Host "  • docker-compose --profile dev logs gcode-dev         - View container logs" -ForegroundColor White
Write-Host "  • docker-compose --profile dev down                   - Stop container" -ForegroundColor White
Write-Host ""
Write-Host "💡 To start using gcode in the container:" -ForegroundColor Yellow
Write-Host "   docker-compose --profile dev exec gcode-dev gcode" -ForegroundColor White
Write-Host ""
Write-Host "🎯 The container has access to:" -ForegroundColor Yellow
Write-Host "   • Your current project directory (mounted at /app)" -ForegroundColor White
Write-Host "   • Git configuration" -ForegroundColor White
Write-Host "   • SSH keys (if available)" -ForegroundColor White
Write-Host "   • gcode configuration in .gcode/" -ForegroundColor White

# Optionally enter the container
if ($EnterContainer) {
    Write-Host "🔌 Entering gcode development container..." -ForegroundColor Green
    docker-compose --profile dev exec gcode-dev bash
} else {
    $response = Read-Host "🤔 Would you like to enter the container now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "🔌 Entering gcode development container..." -ForegroundColor Green
        docker-compose --profile dev exec gcode-dev bash
    }
}
