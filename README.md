# gcode

Your intelligent coding companion - like Claude Code but with dual API support and real collapsible sections.

## 🚀 Quick Start

### Install globally (like Claude Code):
```bash
# From the gcode directory
pip install -e .

# Now use gcode anywhere!
gcode
```

### Or run directly:
```bash
python -m gcode.agent
```

## 🐳 Docker Support

### Quick Docker Usage:
```bash
# Build and run gcode in Docker
docker build -t gcode .
docker run -it --rm -v $(pwd):/workspace gcode

# Or use docker-compose
docker-compose --profile cli up
```

### Development Container:
```bash
# Start development environment
./run_devcontainer_gcode.sh

# Or on Windows
.\run_devcontainer_gcode.ps1

# Enter the container
docker-compose --profile dev exec gcode-dev bash
```

### VS Code Dev Container:
1. Install "Remote - Containers" extension
2. Open gcode project in VS Code
3. Press `Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
4. VS Code will automatically set up the development environment

## ✨ What gcode does

- **Write, analyze, and refactor code** with natural language
- **Generate tests and documentation** automatically
- **Monitor code quality and security** in real-time
- **Manage git operations** intelligently
- **Provide real-time coding assistance** with collapsible sections

## 🎯 Usage (Just like Claude Code!)

### Interactive Mode (Default):
```bash
gcode
# Enters interactive mode where you can chat naturally
```

### Single Commands:
```bash
gcode 'explain this function'
gcode 'refactor this code for better performance'
gcode 'generate unit tests for the auth module'
gcode 'commit my changes with a descriptive message'
```

## 🔧 Features

### Dual API Support
- **Gemini (Google)** - Free tier, 50 requests/day
- **OpenAI (GPT-4o)** - Premium, unlimited requests
- **Auto-detection** - Automatically finds working API keys

### Real Collapsible Sections
- **Interactive UI** - Expand/collapse sections with commands
- **Professional appearance** - VS Code-inspired design
- **Organized output** - Clean, readable information display

### Advanced Coding Tools
- **Code analysis** - Quality, structure, and improvement suggestions
- **Git integration** - Smart commits, conflict resolution, branching
- **Testing** - Unit tests, property-based tests, security scans
- **Monitoring** - Real-time code quality and performance tracking

### Docker Integration
- **Containerized deployment** - Consistent environment across systems
- **Development containers** - Full VS Code integration
- **Volume mounting** - Access to your project files
- **Git integration** - SSH keys and configuration mounted

## 🎨 Commands

- **`help`** - Show available commands
- **`context`** - Show project insights
- **`demo`** - Demonstrate collapsible sections
- **`toggle`** - Toggle section states
- **`exit`** or **`quit`** - End session

## 🔑 API Setup

### Option 1: Environment Variables
```bash
# For Gemini
export GEMINI_API_KEY='your-gemini-key'

# For OpenAI
export OPENAI_API_KEY='your-openai-key'
```

### Option 2: .env File
Create a `.env` file in your project:
```bash
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

### Option 3: Docker Environment
```bash
# Set in docker-compose.yml
environment:
  - GEMINI_API_KEY=${GEMINI_API_KEY}
  - OPENAI_API_KEY=${OPENAI_API_KEY}
```

## 🚀 Examples

### Code Analysis
```bash
gcode 'analyze this Python file and suggest improvements'
```

### Git Operations
```bash
gcode 'commit my changes with an AI-generated message'
gcode 'create a new feature branch called user-auth'
```

### Testing
```bash
gcode 'generate property-based tests for this function'
gcode 'run a security scan on this codebase'
```

### Documentation
```bash
gcode 'create documentation for this API'
gcode 'explain how this authentication system works'
```

## 🐳 Docker Profiles

### CLI Profile (Default):
```bash
docker-compose --profile cli up
# Runs gcode in interactive mode
```

### Development Profile:
```bash
docker-compose --profile dev up
# Development environment with full tooling
```

### Web Profile:
```bash
docker-compose --profile web up
# Web interface on port 8000
```

## 🎯 Why gcode?

- **Dual API Support** - Use Gemini (free) or OpenAI (premium)
- **Real Collapsible Sections** - Interactive, organized output
- **Professional UI** - Sleek, modern interface
- **Advanced Tools** - Beyond basic code generation
- **Easy Installation** - Works globally like Claude Code
- **Docker Ready** - Containerized deployment and development

## 🔧 Development

```bash
# Clone and install
git clone <your-repo>
cd gcode
pip install -e .

# Run tests
python -m pytest

# Run demo
python test_collapsible.py

# Docker development
./run_devcontainer_gcode.sh
```

## 📄 License

MIT License - see LICENSE file for details.

---

**gcode** - Your intelligent coding companion with the power of dual APIs, real collapsible sections, and Docker support! 🚀🐳
