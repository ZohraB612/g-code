import openai
import google.generativeai as genai
import os
import sys
import json
import time
import threading
import tty
import termios
import select
from pathlib import Path
from datetime import datetime
from .tools import (
    read_file, 
    write_file, 
    run_shell_command, 
    get_project_structure,
    analyze_python_file,
    create_test_file,
    search_code,
    install_dependencies,
    run_tests,
    # Advanced Git Integration
    git_status,
    git_commit_with_ai_message,
    git_resolve_conflicts,
    git_smart_branch,
    # Real-time Code Monitoring
    monitor_code_quality_continuous,
    auto_fix_common_issues,
    # Advanced Testing
    generate_property_based_tests,
    run_security_scan,
    performance_profiling
)
from dotenv import load_dotenv
import pickle
from .file_watcher import FileWatcher, create_file_watcher

load_dotenv()

# Professional color scheme (VS Code/Cursor inspired)
class Colors:
    PRIMARY = '\033[38;5;33m'      # Blue
    SECONDARY = '\033[38;5;240m'   # Gray
    SUCCESS = '\033[38;5;34m'      # Green
    WARNING = '\033[38;5;208m'     # Orange
    ERROR = '\033[38;5;196m'        # Red
    INFO = '\033[38;5;39m'         # Light Blue
    HIGHLIGHT = '\033[38;5;220m'   # Yellow
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def colored(text, color, bold=False):
    """Apply professional color to text."""
    if sys.platform != 'win32' and os.isatty(sys.stdout.fileno()):
        style = color + (Colors.BOLD if bold else '')
        return f"{style}{text}{Colors.RESET}"
    return text

class APIAuthenticator:
    """Handles API authentication and selection between Gemini and OpenAI."""
    
    def __init__(self):
        self.config_file = Path(".gcode_config.json")
        self.selected_api = None
        self.api_key = None
        self.model_name = None
        
    def show_welcome(self):
        """Show welcome message and API selection menu."""
        print(f"\n{colored('gcode', Colors.HIGHLIGHT, bold=True)}")
        print(f"{colored('Your intelligent coding companion', Colors.SECONDARY)}")
        print(f"{colored('Choose your AI service:', Colors.PRIMARY, bold=True)}\n")
        
    def select_api(self):
        """Let user select between Gemini and OpenAI."""
        print(f"{colored('Available Services:', Colors.INFO)}")
        print(f"  1. {colored('Gemini', Colors.PRIMARY)} - Free, 50 requests/day")
        print(f"  2. {colored('OpenAI', Colors.SUCCESS)} - Premium, unlimited")
        print(f"  3. {colored('Auto-detect', Colors.HIGHLIGHT)} - Use available keys\n")
        
        while True:
            try:
                choice = input(f"{colored('Select (1-3): ', Colors.PRIMARY)}").strip()
                if choice == '1':
                    return 'gemini'
                elif choice == '2':
                    return 'openai'
                elif choice == '3':
                    return 'auto'
                else:
                    print(f"{colored('Invalid choice. Please enter 1, 2, or 3.', Colors.ERROR)}")
            except KeyboardInterrupt:
                print(f"\n{colored('Setup cancelled. Goodbye!', Colors.WARNING)}")
                sys.exit(0)
    
    def get_api_key(self, api_type):
        """Get API key from user or environment."""
        if api_type == 'gemini':
            env_key = os.getenv("GEMINI_API_KEY")
            if env_key:
                print(f"{colored('✅ Found GEMINI_API_KEY in environment', Colors.SUCCESS)}")
                return env_key
            
            print(f"\n{colored('🔑 Gemini API Setup:', Colors.INFO)}")
            print(f"  1. Visit: {colored('https://makersuite.google.com/app/apikey', Colors.PRIMARY)}")
            print(f"  2. Create a new API key")
            print(f"  3. Copy the key (starts with 'AIza...')\n")
            
        elif api_type == 'openai':
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                print(f"{colored('✅ Found OPENAI_API_KEY in environment', Colors.SUCCESS)}")
                return env_key
            
            print(f"\n{colored('🔑 OpenAI API Setup:', Colors.INFO)}")
            print(f"  1. Visit: {colored('https://platform.openai.com/api-keys', Colors.PRIMARY)}")
            print(f"  2. Create a new API key")
            print(f"  3. Copy the key (starts with 'sk-...')\n")
        
        while True:
            try:
                api_key = input(f"{colored('Enter your API key: ', Colors.PRIMARY)}").strip()
                if api_key:
                    return api_key
                else:
                    print(f"{colored('API key cannot be empty.', Colors.ERROR)}")
            except KeyboardInterrupt:
                print(f"\n{colored('Setup cancelled. Goodbye!', Colors.WARNING)}")
                sys.exit(0)
    
    def test_api(self, api_type, api_key):
        """Test the API key to ensure it works."""
        print(f"{colored('Testing connection...', Colors.INFO)}")
        
        try:
            if api_type == 'gemini':
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                response = model.generate_content("Say 'Hello from Gemini!'")
                if response.text:
                    print(f"{colored('✓', Colors.SUCCESS)} Gemini ready")
                    return True
                    
            elif api_type == 'openai':
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'Hello from OpenAI!'"}],
                    max_tokens=10
                )
                if response.choices[0].message.content:
                    print(f"{colored('✓', Colors.SUCCESS)} OpenAI ready")
                    return True
                    
        except Exception as e:
            print(f"{colored('✗', Colors.ERROR)} Connection failed")
            return False
        
        return False
    
    def save_config(self, api_type, api_key):
        """Save the selected API configuration."""
        config = {
            'selected_api': api_type,
            'api_key': api_key,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"{colored('✓', Colors.SUCCESS)} Configuration saved")
        except Exception as e:
            print(f"{colored('⚠', Colors.WARNING)} Could not save config")
    
    def load_config(self):
        """Load saved configuration if available."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Test if the saved API still works
                if self.test_api(config['selected_api'], config['api_key']):
                    self.selected_api = config['selected_api']
                    self.api_key = config['api_key']
                    print(f"{colored('✓', Colors.SUCCESS)} Using saved configuration")
                    return True
                else:
                    print(f"{colored('⚠', Colors.WARNING)} Saved configuration expired")
                    
            except Exception as e:
                print(f"{colored('⚠', Colors.WARNING)} Could not load saved configuration")
        
        return False
    
    def auto_detect(self):
        """Automatically detect which API keys are available and working."""
        print(f"{colored('Auto-detecting...', Colors.INFO)}")
        
        # Check Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key and self.test_api('gemini', gemini_key):
            print(f"{colored('✓', Colors.SUCCESS)} Auto-selected Gemini")
            return 'gemini', gemini_key
        
        # Check OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and self.test_api('openai', openai_key):
            print(f"{colored('✓', Colors.SUCCESS)} Auto-selected OpenAI")
            return 'openai', openai_key
        
        print(f"{colored('✗', Colors.ERROR)} No working keys found")
        return None, None
    
    def authenticate(self):
        """Main authentication flow."""
        # Try to load saved config first
        if self.load_config():
            return self.selected_api, self.api_key
        
        # Show welcome and get user choice
        self.show_welcome()
        choice = self.select_api()
        
        if choice == 'auto':
            api_type, api_key = self.auto_detect()
            if api_type:
                self.save_config(api_type, api_key)
                return api_type, api_key
            else:
                print(f"{colored('Please set up an API key manually.', Colors.INFO)}")
                choice = self.select_api()
        
        # Get API key for selected service
        api_key = self.get_api_key(choice)
        
        # Test the API
        if self.test_api(choice, api_key):
            self.save_config(choice, api_key)
            return choice, api_key
        else:
            print(f"{colored('❌ Authentication failed. Please check your API key.', Colors.ERROR)}")
            return None, None

class InteractiveTerminal:
    """Handles interactive terminal input for collapsible sections."""
    
    def __init__(self):
        self.old_settings = None
        
    def __enter__(self):
        """Enter interactive mode."""
        if sys.platform != 'win32':
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit interactive mode."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Get a single key press."""
        if sys.platform == 'win32':
            import msvcrt
            return msvcrt.getch().decode('utf-8')
        else:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

class ProfessionalUI:
    """Professional UI system with REAL collapsible sections and interactive formatting."""
    
    def __init__(self):
        self.section_id = 0
        self.sections = {}
        self.active_sections = set()
        self.section_content = {}
        self.terminal = InteractiveTerminal()
    
    def section(self, title, content="", collapsible=True, expanded=False):
        """Create a collapsible section with professional styling."""
        self.section_id += 1
        section_id = self.section_id
        
        # Store section info
        self.sections[section_id] = {
            'title': title,
            'content': content,
            'collapsible': collapsible,
            'expanded': expanded
        }
        
        if expanded:
            self.active_sections.add(section_id)
        
        # Store content for later rendering
        self.section_content[section_id] = content
        
        # Return the section header
        if collapsible:
            status = "▼" if expanded else "▶"
            header = f"{status} {title}"
            if expanded:
                self.active_sections.add(section_id)
                return f"\n{colored(header, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}\n{content}"
            else:
                return f"\n{colored(header, Colors.PRIMARY)}\n{colored('─' * len(title), Colors.SECONDARY)}"
        else:
            return f"\n{colored(title, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}\n{content}"
    
    def render_section(self, section_id, force_expand=False):
        """Render a section with current state."""
        if section_id not in self.sections:
            return ""
        
        section = self.sections[section_id]
        title = section['title']
        content = section['content']
        collapsible = section['collapsible']
        
        if force_expand or section_id in self.active_sections:
            if collapsible:
                header = f"▼ {title}"
                return f"\n{colored(header, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}\n{content}"
            else:
                return f"\n{colored(title, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}\n{content}"
        else:
            if collapsible:
                header = f"▶ {title}"
                return f"\n{colored(header, Colors.PRIMARY)}\n{colored('─' * len(title), Colors.SECONDARY)}"
            else:
                return f"\n{colored(title, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}\n{content}"
    
    def toggle_section(self, section_id):
        """Toggle a section's expanded state."""
        if section_id in self.sections and self.sections[section_id]['collapsible']:
            if section_id in self.active_sections:
                self.active_sections.remove(section_id)
            else:
                self.active_sections.add(section_id)
            return True
        return False
    
    def interactive_render(self, prompt="Press SPACE to toggle sections, ENTER to continue, Q to quit"):
        """Render all sections interactively with user control."""
        if not self.sections:
            return
        
        print(f"\n{colored('Interactive Sections:', Colors.HIGHLIGHT, bold=True)}")
        print(colored(prompt, Colors.SECONDARY))
        
        try:
            # Simple interactive mode without complex terminal handling
            while True:
                # Clear screen and re-render
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Show current state
                print(f"{colored('Interactive Sections:', Colors.HIGHLIGHT, bold=True)}")
                print(colored(prompt, Colors.SECONDARY))
                
                # Render all sections
                for section_id in sorted(self.sections.keys()):
                    print(self.render_section(section_id))
                
                # Show controls
                print(f"\n{colored('Controls:', Colors.INFO)}")
                print(f"  {colored('SPACE', Colors.PRIMARY)} - Toggle section")
                print(f"  {colored('ENTER', Colors.PRIMARY)} - Continue")
                print(f"  {colored('Q', Colors.PRIMARY)} - Quit")
                print(f"  {colored('1-9', Colors.PRIMARY)} - Toggle specific section")
                
                # Get user input
                try:
                    user_input = input(f"\n{colored('Command: ', Colors.PRIMARY)}").strip().lower()
                    
                    if user_input == ' ' or user_input == 'space':
                        # Toggle first collapsible section
                        for section_id in sorted(self.sections.keys()):
                            if self.sections[section_id]['collapsible']:
                                self.toggle_section(section_id)
                                break
                    elif user_input == '' or user_input == 'enter':
                        break
                    elif user_input == 'q' or user_input == 'quit':
                        return False
                    elif user_input.isdigit():
                        # Toggle specific section by number
                        section_num = int(user_input)
                        section_ids = sorted(self.sections.keys())
                        if 1 <= section_num <= len(section_ids):
                            section_id = section_ids[section_num - 1]
                            if self.sections[section_id]['collapsible']:
                                self.toggle_section(section_id)
                    elif user_input == 'help':
                        print(f"\n{colored('Section Commands:', Colors.INFO)}")
                        print(f"  {colored('SPACE', Colors.PRIMARY)} - Toggle first collapsible section")
                        print(f"  {colored('1-9', Colors.PRIMARY)} - Toggle section by number")
                        print(f"  {colored('ENTER', Colors.PRIMARY)} - Continue to main CLI")
                        print(f"  {colored('Q', Colors.PRIMARY)} - Quit interactive mode")
                        input(f"\n{colored('Press ENTER to continue...', Colors.SECONDARY)}")
                        
                except KeyboardInterrupt:
                    print(f"\n{colored('Interactive mode interrupted. Goodbye!', Colors.WARNING)}")
                    return False
                    
        except Exception as e:
            print(f"Interactive mode error: {e}")
        
        return True
    
    def subsection(self, title):
        """Create a subsection with subtle styling."""
        return f"\n{colored('  ' + title, Colors.SECONDARY, bold=True)}"
    
    def info(self, text):
        """Display informational text."""
        return colored(f"ℹ  {text}", Colors.INFO)
    
    def success(self, text):
        """Display success message."""
        return colored(f"✓ {text}", Colors.SUCCESS)
    
    def warning(self, text):
        """Display warning message."""
        return colored(f"⚠  {text}", Colors.WARNING)
    
    def error(self, text):
        """Display error message."""
        return colored(f"✗ {text}", Colors.ERROR)
    
    def progress_bar(self, current, total, width=40):
        """Display a professional progress bar."""
        if total == 0:
            return ""
        
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(100 * current / total)
        
        return f"[{bar}] {percentage}% ({current}/{total})"
    
    def spinner(self, text="Processing"):
        """Display a spinning progress indicator."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return f"{spinner_chars[int(time.time() * 10) % len(spinner_chars)]} {text}"
    
    def file_preview(self, file_path, max_lines=10):
        """Display a file preview with syntax highlighting."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            preview = f"\n{colored('File Preview:', Colors.HIGHLIGHT, bold=True)} {file_path}\n"
            preview += colored('─' * 50, Colors.SECONDARY) + "\n"
            
            for i, line in enumerate(lines[:max_lines], 1):
                preview += f"{colored(f'{i:3d}', Colors.SECONDARY)} {line.rstrip()}\n"
            
            if len(lines) > max_lines:
                preview += f"{colored('...', Colors.SECONDARY)} ({len(lines) - max_lines} more lines)\n"
            
            return preview
        except Exception:
            return f"\n{colored('Could not preview file:', Colors.ERROR)} {file_path}\n"

# Define the available tools for the agent
tools = [
    read_file, 
    write_file, 
    run_shell_command, 
    get_project_structure,
    analyze_python_file,
    create_test_file,
    search_code,
    install_dependencies,
    run_tests,
    # Advanced Git Integration
    git_status,
    git_commit_with_ai_message,
    git_resolve_conflicts,
    git_smart_branch,
    # Real-time Code Monitoring
    monitor_code_quality_continuous,
    auto_fix_common_issues,
    # Advanced Testing
    generate_property_based_tests,
    run_security_scan,
    performance_profiling
]

# A mapping of tool names to their functions for easy lookup
AVAILABLE_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_command": run_shell_command,
    "get_project_structure": get_project_structure,
    "analyze_python_file": analyze_python_file,
    "create_test_file": create_test_file,
    "search_code": search_code,
    "install_dependencies": install_dependencies,
    "run_tests": run_tests,
    # Advanced Git Integration
    "git_status": git_status,
    "git_commit_with_ai_message": git_commit_with_ai_message,
    "git_resolve_conflicts": git_resolve_conflicts,
    "git_smart_branch": git_smart_branch,
    # Real-time Code Monitoring
    "monitor_code_quality_continuous": monitor_code_quality_continuous,
    "auto_fix_common_issues": auto_fix_common_issues,
    # Advanced Testing
    "generate_property_based_tests": generate_property_based_tests,
    "run_security_scan": run_security_scan,
    "performance_profiling": performance_profiling
}

# Convert our tools to OpenAI function format
def create_openai_tools():
    """Convert our tools to OpenAI function calling format for v0.28.1."""
    openai_functions = []
    
    tool_descriptions = {
        "read_file": "Reads the content of a file",
        "write_file": "Writes content to a file, creating it if it doesn't exist",
        "run_shell_command": "Executes a shell command and returns its output",
        "get_project_structure": "Returns a string representing the project's file structure",
        "analyze_python_file": "Analyzes a Python file for code quality, structure, and potential improvements",
        "create_test_file": "Creates a basic test file for a given Python source file",
        "search_code": "Searches for code patterns, functions, or text across Python files in a directory",
        "install_dependencies": "Installs Python dependencies from a requirements file",
        "run_tests": "Runs Python tests in the specified directory",
        "git_status": "Gets the current git status with detailed information about changes",
        "git_commit_with_ai_message": "Commits changes with an AI-generated commit message based on the changes",
        "git_resolve_conflicts": "Attempts to automatically resolve git merge conflicts",
        "git_smart_branch": "Performs intelligent git branch operations with safety checks",
        "monitor_code_quality_continuous": "Continuously monitors code quality and provides real-time feedback",
        "auto_fix_common_issues": "Automatically fixes common code quality issues",
        "generate_property_based_tests": "Generates property-based tests using hypothesis for a Python file",
        "run_security_scan": "Runs a basic security scan on Python files for common vulnerabilities",
        "performance_profiling": "Performs basic performance profiling on Python code"
    }
    
    for tool_name in AVAILABLE_TOOLS.keys():
        tool_func = AVAILABLE_TOOLS[tool_name]
        
        # Get function signature
        import inspect
        sig = inspect.signature(tool_func)
        
        # Create OpenAI function definition for v0.28.1
        func_def = {
            "name": tool_name,
            "description": tool_descriptions.get(tool_name, f"Executes {tool_name}"),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Add parameters based on function signature
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int:
                    param_type = "integer"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == float:
                    param_type = "number"
            
            func_def["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name} for {tool_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                func_def["parameters"]["required"].append(param_name)
        
        openai_functions.append(func_def)
    
    return openai_functions

class ProjectContext:
    """Manages project context and memory across sessions."""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.context_file = self.project_root / ".gcode_context.json"
        self.conversation_history = []
        self.project_insights = {}
        self.last_analysis = None
        self.load_context()
    
    def load_context(self):
        """Load existing context from file."""
        if self.context_file.exists():
            try:
                with open(self.context_file, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = data.get('conversation_history', [])
                    self.project_insights = data.get('project_insights', {})
                    self.last_analysis = data.get('last_analysis')
            except Exception as e:
                print(colored(f"Warning: Could not load context: {e}", Colors.WARNING))
    
    def save_context(self):
        """Save current context to file."""
        try:
            data = {
                'conversation_history': self.conversation_history[-50:],  # Keep last 50 interactions
                'project_insights': self.project_insights,
                'last_analysis': self.last_analysis,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(colored(f"Warning: Could not save context: {e}", Colors.WARNING))
    
    def add_interaction(self, user_input, agent_response, tools_used):
        """Add a new interaction to the conversation history."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'agent_response': agent_response,
            'tools_used': tools_used,
            'project_state': self._capture_project_state()
        }
        self.conversation_history.append(interaction)
        self.save_context()
    
    def _capture_project_state(self):
        """Capture current project state for context."""
        try:
            return {
                'files': [str(f) for f in list(self.project_root.rglob('*.py'))[:20]],  # Convert Path to string
                'structure': get_project_structure(),
                'working_directory': str(os.getcwd())
            }
        except Exception:
            return {}
    
    def get_relevant_context(self, current_request):
        """Get relevant context for the current request."""
        relevant = []
        
        # Add recent relevant conversations
        for interaction in reversed(self.conversation_history[-5:]):
            if any(keyword in interaction['user_input'].lower() 
                   for keyword in current_request.lower().split()):
                relevant.append(interaction)
        
        # Add project insights
        if self.project_insights:
            relevant.append({'type': 'project_insights', 'data': self.project_insights})
        
        return relevant

class GeminiAgent:
    """A conversational agent powered by either Gemini or OpenAI with advanced capabilities."""
    
    def __init__(self, model_name=None):
        """Initializes the agent with API authentication and advanced context management."""
        # Authenticate and configure API
        self.authenticator = APIAuthenticator()
        self.api_type, self.api_key = self.authenticator.authenticate()
        
        if not self.api_type or not self.api_key:
            print(f"{colored('❌ Authentication failed. Cannot proceed.', Colors.ERROR)}")
            sys.exit(1)
        
        # Configure the selected API
        self._configure_api()
        
        # Set model name based on API type
        if not model_name:
            if self.api_type == 'gemini':
                model_name = "gemini-1.5-flash-latest"
            else:  # openai
                model_name = "gpt-4o"
        
        self.model_name = model_name
        self.context = ProjectContext()
        self.ui = ProfessionalUI()
        
        # Initialize API-specific components
        if self.api_type == 'openai':
            self.openai_functions = create_openai_tools()
        else:  # gemini
            self.openai_functions = None
        
        self.conversation_history = []
        
        # Enhanced system prompt for advanced capabilities
        self.system_prompt = """You are an expert software engineer and pair-programmer, similar to Cursor or Claude Code. 
Your goal is to help with coding projects efficiently and conversationally. 
You have access to tools to read/write files, run commands, and analyze project structure. 

Key principles:
1. Be proactive and autonomous - use tools when needed without asking permission
2. Provide clear, actionable plans before executing
3. Think step-by-step and explain your reasoning
4. Be conversational and helpful, not robotic
5. When you need to use tools, explain what you're doing and why
6. Always provide a summary of what was accomplished
7. IMPORTANT: Provide ALL necessary tool calls in a single response to complete the entire task
8. Don't stop after one tool - think through the complete workflow and provide all tools needed
9. Analyze the project context and suggest improvements proactively
10. Remember previous work and build upon it intelligently
11. Handle edge cases and errors gracefully
12. Suggest next steps and improvements after completing tasks

Remember: You're a coding partner, not just a tool executor. Complete the entire task in one go and be proactive about helping improve the codebase."""
        
        # Initialize with project analysis
        self._analyze_project_context()
        
        # Initialize file watcher
        self.file_watcher = None
        self.watching_files = False
        self.auto_analysis = True
        self.auto_watch_files = True  # New: control auto-file-watching
        
        # Automatically start file watching for agentic behavior
        if self.auto_watch_files:
            self._auto_start_file_watching()
    
    def _configure_api(self):
        """Configure the selected API with the provided key."""
        if self.api_type == 'gemini':
            genai.configure(api_key=self.api_key)
            print(f"{colored('✓', Colors.SUCCESS)} Using Gemini")
        elif self.api_type == 'openai':
            openai.api_key = self.api_key
            print(f"{colored('✓', Colors.SUCCESS)} Using OpenAI")
    
    def _call_api(self, prompt):
        """Call the configured API (Gemini or OpenAI) with the prompt and tools."""
        try:
            if self.api_type == 'gemini':
                return self._call_gemini(prompt)
            else:  # openai
                return self._call_openai(prompt)
        except Exception as e:
            raise Exception(f"{self.api_type.upper()} API error: {e}")
    
    def _call_gemini(self, prompt):
        """Call Gemini API with the prompt and tools."""
        try:
            model = genai.GenerativeModel(model_name=self.model_name, tools=tools)
            chat = model.start_chat(history=[
                {"role": "user", "parts": [self.system_prompt]}
            ])
            
            response = chat.send_message(prompt)
            
            # Extract the response
            response_parts = response.candidates[0].content.parts
            plan_text = "".join(part.text for part in response_parts if part.text).strip()
            tool_calls = [part.function_call for part in response_parts if part.function_call]
            
            return {
                'content': plan_text,
                'tool_calls': tool_calls
            }
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def _call_openai(self, prompt):
        """Call OpenAI API with the prompt and tools."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Add conversation history
            for conv in self.conversation_history[-5:]:  # Last 5 conversations
                messages.append({"role": "user", "content": conv['user_input']})
                messages.append({"role": "assistant", "content": conv['agent_response']})
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                functions=self.openai_functions,
                function_call="auto"
            )
            
            # Extract the response
            assistant_message = response.choices[0].message
            
            return {
                'content': assistant_message.content or "Tool execution completed",
                'tool_calls': [assistant_message.function_call] if hasattr(assistant_message, 'function_call') and assistant_message.function_call else []
            }
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    def _enhance_prompt_with_context(self, prompt, relevant_context):
        """Enhance the user prompt with relevant context."""
        if not relevant_context:
            return prompt
        
        context_info = "\n\nRelevant context from previous work:\n"
        for i, ctx in enumerate(relevant_context[-3:], 1):  # Last 3 relevant items
            if 'user_input' in ctx:
                context_info += f"{i}. Previous request: {ctx['user_input']}\n"
                if 'tools_used' in ctx:
                    context_info += f"   Tools used: {[t['name'] for t in ctx['tools_used']]}\n"
        
        context_info += f"\nCurrent project insights: {self.context.project_insights}\n"
        
        return prompt + context_info

    def _provide_proactive_suggestions(self, original_request, tools_used):
        """Provide proactive suggestions based on the completed work."""
        try:
            suggestions_prompt = f"""
Based on the completed request: "{original_request}"
And the tools used: {[t['name'] for t in tools_used]}

What would be helpful next steps or improvements? Consider:
1. Code quality improvements
2. Testing suggestions
3. Documentation needs
4. Performance optimizations
5. Next logical development steps

Provide 2-3 specific, actionable suggestions.
"""
            
            response = self._call_api(suggestions_prompt)
            suggestions = response.get('content', '')
            
            if suggestions:
                print(self.ui.section("Proactive Suggestions", suggestions, collapsible=True, expanded=False))
                
        except Exception as e:
            print(self.ui.warning(f"Could not generate suggestions: {e}"))

    def _check_for_more_work(self, original_prompt, executed_plan):
        """Check if we need to continue with more tools to complete the task."""
        # If the plan mentioned multiple steps but we only executed one tool,
        # ask the model to continue
        if "step" in executed_plan.lower() and "step 1" in executed_plan.lower():
            print(self.ui.info("Checking if more work is needed..."))
            
            try:
                follow_up_prompt = f"Continue with the remaining steps from the plan: {executed_plan}"
                response = self._call_api(follow_up_prompt)
                
                tool_calls = response.get('tool_calls', [])
                
                if tool_calls:
                    print(self.ui.subsection(f"Executing {len(tool_calls)} additional tool(s)"))
                    for i, tool_call in enumerate(tool_calls, 1):
                        self._execute_tool(tool_call, i, len(tool_calls))
                    
                    print(self.ui.success("Additional work completed"))
                
            except Exception as e:
                print(self.ui.warning(f"Could not check for additional work: {e}"))

    def _execute_tool(self, tool_call, current, total):
        """Execute a single tool call with professional feedback."""
        func_name = tool_call['name']
        func_to_call = AVAILABLE_TOOLS.get(func_name)
        
        if not func_to_call:
            print(self.ui.error(f"Unknown tool '{func_name}'"))
            return None
        
        # Parse arguments from OpenAI function call (older format)
        import json
        try:
            func_args = json.loads(tool_call.get('arguments', '{}'))
        except:
            func_args = {}
        
        # Show progress and tool info
        print(f"\n{colored(f'[{current}/{total}]', Colors.PRIMARY)} {func_name}")
        if func_args:
            print(f"   Args: {func_args}")
        
        try:
            result = func_to_call(**func_args)
            print(f"   {colored('Result:', Colors.SUCCESS)} {result}")
            
            # Return tool execution info for context
            return {
                'name': func_name,
                'args': func_args,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"   {colored('Error:', Colors.ERROR)} {str(e)}")
            return None

    def _show_help(self):
        """Show available commands and help."""
        help_text = f"""
Available Commands:
- Type your request normally (e.g., "create a new file called main.py")
- 'help' - Show this help message
- 'context' - Show project insights
- 'interactive' - Enter collapsible section mode
- 'toggle' - Toggle collapsible sections
- 'demo' - Demonstrate collapsible sections
- 'exit' or 'quit' - End the session

Examples:
- "Create a Python file with a hello world function"
- "Show me the project structure"
- "Read the contents of agent.py"
- "Run 'ls -la' to see files"
- "Analyze this code and suggest improvements"
- "Create a test file for the existing functions"

Advanced Features (Beyond Cursor/Claude Code):

Git Integration:
- "Commit my changes with an AI-generated message"
- "Create a new feature branch called 'user-auth'"
- "Resolve merge conflicts automatically"
- "Show git status and recent changes"

Real-time Monitoring:
- "Monitor code quality across the project"
- "Auto-fix common code quality issues"
- "Check for performance bottlenecks"
- "Scan for security vulnerabilities"

Advanced Testing:
- "Generate property-based tests for agent.py"
- "Run security vulnerability scan"
- "Profile performance of the codebase"
- "Create comprehensive test suites"

Pro Tips:
- "Help me refactor this code for better performance"
- "Suggest architectural improvements for this project"
- "Automate my development workflow"
- "Set up CI/CD pipeline for this project"
        """
        print(colored(help_text, Colors.INFO))

    def _show_project_context(self):
        """Show current project context and insights."""
        insights = self.context.project_insights
        
        context_content = ""
        if insights:
            context_content += f"Python Files: {insights.get('total_python_files', 0)}\n"
            if insights.get('main_files'):
                context_content += f"Main Files: {', '.join(insights['main_files'])}\n"
            
            context_content += f"Requirements: {'Yes' if insights.get('has_requirements') else 'No'}\n"
            context_content += f"Setup Files: {'Yes' if insights.get('has_setup') else 'No'}\n"
            
            if insights.get('last_analysis'):
                context_content += f"Last Analyzed: {insights['last_analysis'][:19]}\n"
        else:
            context_content += "No project insights available yet.\n"
        
        # Show recent conversation history
        if self.context.conversation_history:
            context_content += f"\nRecent Conversations: {len(self.context.conversation_history)}\n"
            for i, conv in enumerate(self.context.conversation_history[-3:], 1):
                context_content += f"   {i}. {conv['user_input'][:50]}...\n"
        
        print(self.ui.section("Project Context & Insights", context_content, collapsible=True, expanded=False))

    def _analyze_project_context(self):
        """Analyze the project context to understand the codebase."""
        try:
            # Create collapsible section for project analysis
            analysis_content = self.ui.info("Analyzing project context...")
            
            # Get project structure
            structure = get_project_structure()
            
            # Analyze Python files for insights
            python_files = list(Path(".").rglob("*.py"))
            insights = {
                'total_python_files': len(python_files),
                'main_files': [f.name for f in python_files if f.name in ['main.py', 'app.py', '__main__.py']],
                'has_requirements': Path("requirements.txt").exists(),
                'has_setup': Path("setup.py").exists() or Path("pyproject.toml").exists(),
                'project_structure': structure
            }
            
            self.context.project_insights = insights
            self.context.last_analysis = datetime.now().isoformat()
            self.context.save_context()
            
            success_msg = self.ui.success(f"Project analyzed: {insights['total_python_files']} Python files found")
            
            # Create the section with content
            print(self.ui.section("Project Analysis", analysis_content + "\n" + success_msg, collapsible=True, expanded=True))
            
        except Exception as e:
            print(self.ui.error(f"Could not analyze project: {e}"))
    
    def converse(self, prompt: str, interactive=False):
        """Handles the conversation flow with advanced context awareness - like Claude Code."""
        if interactive:
            print(self.ui.section("Interactive Mode", collapsible=False))
            print(self.ui.info("Welcome to gcode! I'm your AI coding companion."))
            print(self.ui.info("I understand your codebase and can help you code faster."))
            print()
            print(self.ui.info("Commands:"))
            print(self.ui.info("  • Type natural language requests (e.g., 'explain this function')"))
            print(self.ui.info("  • Type 'help' for available commands"))
            print(self.ui.info("  • Type 'context' to see project insights"))
            print(self.ui.info("  • Type 'demo' to see collapsible sections"))
            print(self.ui.info("  • Type 'exit' or 'quit' to end the session"))
            print(colored("─" * 60, Colors.SECONDARY))
            
            while True:
                try:
                    user_input = input(colored("\n💬 You: ", Colors.PRIMARY)).strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        print(self.ui.info("Session ended. Happy coding!"))
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'context':
                        self._show_project_context()
                        continue
                    elif user_input.lower() == 'demo':
                        self._demo_collapsible_sections()
                        continue
                    elif user_input.lower() == 'toggle':
                        self._toggle_sections()
                        continue
                    elif not user_input:
                        continue
                    
                    # Process natural language request like Claude Code
                    print(colored(f"\n🤖 gcode: ", Colors.SUCCESS) + colored("Processing your request...", Colors.INFO))
                    self._process_request(user_input)
                    
                except KeyboardInterrupt:
                    print(self.ui.info("Session interrupted. Goodbye!"))
                    break
                except EOFError:
                    print(self.ui.info("End of input. Goodbye!"))
                    break
        else:
            # Single request mode - like Claude Code
            self._process_request(prompt)

    def start_file_watching(self, project_path: str = None):
        """Start watching for file changes and automatically analyze them."""
        if self.watching_files:
            print("⚠️  File watching is already active.")
            return
            
        if project_path is None:
            project_path = os.getcwd()
            
        try:
            # Create file watcher with callback to gcode analysis
            self.file_watcher = create_file_watcher(project_path, self._on_file_change)
            self.file_watcher.start()
            self.watching_files = True
            
            # Only show verbose output if manually started
            if not hasattr(self, '_auto_started'):
                print(f"🔍 File watching started for: {project_path}")
                print("📝 Any file changes will automatically trigger analysis.")
                print("💡 Use 'watch stop' to stop watching, 'watch status' for info.")
            
        except Exception as e:
            print(f"❌ Failed to start file watching: {e}")
            print("💡 Make sure 'watchdog' is installed: pip install watchdog")
    
    def stop_file_watching(self):
        """Stop watching for file changes."""
        if not self.watching_files or self.file_watcher is None:
            print("⚠️  File watching is not active.")
            return
            
        self.file_watcher.stop()
        self.file_watcher = None
        self.watching_files = False
        print("🔍 File watching stopped.")
    
    def _on_file_change(self, event_type: str, file_path: str):
        """Callback for file changes - automatically analyze modified files."""
        if not self.auto_analysis:
            return
            
        try:
            # Get relative path for display
            rel_path = os.path.relpath(file_path, os.getcwd())
            
            if event_type == 'modified':
                print(f"\n🔍 File changed: {rel_path}")
                print("🤖 Automatically analyzing changes...")
                
                # Analyze the changed file
                self._auto_analyze_file(file_path)
                
            elif event_type == 'created':
                print(f"\n📁 New file: {rel_path}")
                print("🤖 Analyzing new file...")
                
                # Analyze the new file
                self._auto_analyze_file(file_path)
                
            elif event_type == 'deleted':
                print(f"\n🗑️  File deleted: {rel_path}")
                
        except Exception as e:
            print(f"❌ Error in auto-analysis: {e}")
    
    def _auto_analyze_file(self, file_path: str):
        """Automatically analyze a file when it changes."""
        try:
            # Check if it's a Python file
            if file_path.endswith('.py'):
                print("🐍 Python file detected - running quality analysis...")
                
                # Quick quality check
                result = analyze_python_file(file_path)
                if result:
                    print("✅ Auto-analysis complete!")
                    print(f"📊 Lines: {result.get('lines', 'N/A')}")
                    print(f"🔍 Issues: {result.get('issues', 'N/A')}")
                    
                    # If there are issues, offer to fix them
                    if result.get('issues') and result['issues'] > 0:
                        print("💡 Use 'gcode fix-issues' to automatically fix detected problems.")
            
            # Check if it's a configuration file
            elif any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml']):
                print("⚙️  Configuration file detected - checking syntax...")
                # Could add config validation here
            
            # Check if it's a documentation file
            elif any(file_path.endswith(ext) for ext in ['.md', '.txt', '.rst']):
                print("📚 Documentation file detected - checking formatting...")
                # Could add markdown linting here
                
        except Exception as e:
            print(f"⚠️  Auto-analysis failed: {e}")
    
    def _auto_start_file_watching(self):
        """Automatically start file watching for proactive monitoring."""
        try:
            # Only auto-start if we're in a project directory (has Python files, git, etc.)
            if self._should_auto_watch():
                self.start_file_watching()
                
                # Mark as auto-started to avoid duplicate messages
                self._auto_started = True
                
                # Show a subtle notification
                print("🔍 Auto-watching files for proactive analysis...")
                print("💡 Use 'watch stop' to disable, 'watch status' for info")
            else:
                print("💡 Not auto-watching (not in a project directory)")
                print("💡 Use 'watch start' to manually begin monitoring")
                
        except Exception as e:
            # Don't fail initialization if file watching fails
            print(f"⚠️  Auto-file-watching failed: {e}")
            print("💡 You can still manually start with 'watch start'")
    
    def _should_auto_watch(self) -> bool:
        """Determine if we should automatically start file watching."""
        current_dir = Path.cwd()
        
        # Check for project indicators
        project_indicators = [
            current_dir / '.git',           # Git repository
            current_dir / 'requirements.txt', # Python project
            current_dir / 'package.json',   # Node.js project
            current_dir / 'Cargo.toml',     # Rust project
            current_dir / 'pom.xml',        # Java/Maven project
            current_dir / 'build.gradle',   # Java/Gradle project
            current_dir / 'Makefile',       # C/C++ project
            current_dir / 'CMakeLists.txt', # CMake project
        ]
        
        # Check if any project indicators exist
        has_project_files = any(indicator.exists() for indicator in project_indicators)
        
        # Also check for source code files
        has_source_files = any(
            current_dir.glob(f"*.{ext}") 
            for ext in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'rs', 'go']
        )
        
        return has_project_files or has_source_files
    
    def toggle_auto_watch(self):
        """Toggle automatic file watching on/off."""
        self.auto_watch_files = not self.auto_watch_files
        status = "enabled" if self.auto_watch_files else "disabled"
        print(f"🔍 Auto-file-watching {status}.")
        
        # If enabling and not currently watching, start now
        if self.auto_watch_files and not self.watching_files:
            self._auto_start_file_watching()
        # If disabling and currently watching, stop now
        elif not self.auto_watch_files and self.watching_files:
            self.stop_file_watching()
    
    def get_watch_status(self):
        """Get current file watching status."""
        if not self.watching_files or self.file_watcher is None:
            return {
                'active': False,
                'message': 'File watching is not active'
            }
        
        status = self.file_watcher.get_status()
        return {
            'active': True,
            'project_path': status['project_path'],
            'files_monitored': status['files_monitored'],
            'auto_analysis': self.auto_analysis
        }
    
    def toggle_auto_analysis(self):
        """Toggle automatic analysis on/off."""
        self.auto_analysis = not self.auto_analysis
        status = "enabled" if self.auto_analysis else "disabled"
        print(f"🔍 Auto-analysis {status}.")
    
    def watch_commands(self, command: str):
        """Handle file watching commands."""
        if command == 'start':
            self.start_file_watching()
        elif command == 'stop':
            self.stop_file_watching()
        elif command == 'status':
            status = self.get_watch_status()
            if status['active']:
                print(f"🔍 File watching: ACTIVE")
                print(f"📁 Project: {status['project_path']}")
                print(f"📊 Files monitored: {status['files_monitored']}")
                print(f"🤖 Auto-analysis: {'ON' if status['auto_analysis'] else 'OFF'}")
                print(f"🔄 Auto-watch: {'ON' if self.auto_watch_files else 'OFF'}")
            else:
                print(f"🔍 File watching: INACTIVE")
                print(f"💡 Use 'watch start' to begin monitoring")
                print(f"🔄 Auto-watch: {'ON' if self.auto_watch_files else 'OFF'}")
        elif command == 'auto-on':
            self.auto_analysis = True
            print("🤖 Auto-analysis enabled.")
        elif command == 'auto-off':
            self.auto_analysis = False
            print("🤖 Auto-analysis disabled.")
        elif command == 'auto-watch-on':
            self.toggle_auto_watch()
        elif command == 'auto-watch-off':
            self.toggle_auto_watch()
        else:
            print("🔍 File watching commands:")
            print("  watch start         - Start watching for changes")
            print("  watch stop          - Stop watching")
            print("  watch status        - Show current status")
            print("  watch auto-on       - Enable auto-analysis")
            print("  watch auto-off      - Disable auto-analysis")
            print("  watch auto-watch-on - Enable auto-file-watching")
            print("  watch auto-watch-off- Disable auto-file-watching")

    def _enter_interactive_mode(self):
        """Enter the interactive collapsible section mode."""
        print(self.ui.success("🎯 Interactive Mode - Collapsible Sections"))
        print("=" * 50)
        
        # Show available commands
        print("Available Commands:")
        print("- Type your request normally (e.g., 'create a new file called main.py')")
        print("- 'help' - Show this help message")
        print("- 'context' - Show project insights")
        print("- 'interactive' - Enter collapsible section mode")
        print("- 'toggle' - Toggle collapsible sections")
        print("- 'demo' - Demonstrate collapsible sections")
        print("- 'watch start' - Start file watching")
        print("- 'watch stop' - Stop file watching")
        print("- 'watch status' - Show file watching status")
        print("- 'exit' or 'quit' - End the session")
        
        print("\nExamples:")
        print("- 'Create a Python file with a hello world function'")
        print("- 'Show me the project structure'")
        print("- 'Read the contents of agent.py'")
        print("- 'Run ls -la to see files'")
        print("- 'Analyze this code and suggest improvements'")
        print("- 'Start watching for file changes'")
        
        print("\nAdvanced Features (Beyond Cursor/Claude Code):")
        
        print("\nGit Integration:")
        print("- 'Commit my changes with an AI-generated message'")
        print("- 'Create a new feature branch called user-auth'")
        print("- 'Resolve merge conflicts automatically'")
        print("- 'Show git status and recent changes'")
        
        print("\nReal-time Monitoring:")
        print("- 'Monitor code quality across the project'")
        print("- 'Auto-fix common code quality issues'")
        print("- 'Check for performance bottlenecks'")
        print("- 'Scan for security vulnerabilities'")
        
        print("\nFile Watching:")
        print("- 'watch start' - Begin monitoring file changes")
        print("- 'watch stop' - Stop monitoring")
        print("- 'watch status' - Show monitoring status")
        print("- 'watch auto-on' - Enable automatic analysis")
        print("- 'watch auto-off' - Disable automatic analysis")
        
        print("\nAdvanced Testing:")
        print("- 'Generate property-based tests for agent.py'")
        print("- 'Run security vulnerability scan'")
        print("- 'Profile performance of the codebase'")
        print("- 'Create comprehensive test suites'")
        
        print("\nPro Tips:")
        print("- 'Help me refactor this code for better performance'")
        print("- 'Suggest architectural improvements for this project'")
        print("- 'Automate my development workflow'")
        print("- 'Set up CI/CD pipeline for this project'")
        
        # Wait for user input
        input() # Wait for user input
    
    def _toggle_sections(self):
        """Toggle the expanded state of all collapsible sections."""
        for section_id in sorted(self.ui.sections.keys()):
            if self.ui.sections[section_id]['collapsible']:
                self.ui.toggle_section(section_id)
                print(f"Toggled section {section_id}")
            else:
                print(f"Section {section_id} is not collapsible.")

    def _demo_collapsible_sections(self):
        """Demonstrate collapsible sections by creating a few."""
        print(self.ui.info("Demonstrating collapsible sections..."))
        
        # Create a few sections
        self.ui.section("Section 1", "This is the content for Section 1. It can be long and detailed.", collapsible=True, expanded=False)
        self.ui.section("Section 2", "This is the content for Section 2. It's shorter.", collapsible=True, expanded=True)
        self.ui.section("Section 3", "This is the content for Section 3. It's also shorter.", collapsible=True, expanded=False)
        self.ui.section("Non-collapsible Section", "This section cannot be collapsed.", collapsible=False, expanded=True)
        
        # Render them
        print(self.ui.render_section(1))
        print(self.ui.render_section(2))
        print(self.ui.render_section(3))
        print(self.ui.render_section(4))
        
        # Toggle some
        self.ui.toggle_section(1)
        self.ui.toggle_section(3)
        
        print(self.ui.render_section(1))
        print(self.ui.render_section(2))
        print(self.ui.render_section(3))
        print(self.ui.render_section(4))
        
        print(self.ui.info("Press ENTER to continue..."))
        input() # Wait for user input

    def _process_request(self, request: str):
        """Process a user request with enhanced file watching support."""
        # Check for watch commands first
        if request.startswith('watch '):
            command = request[6:].strip()
            self.watch_commands(command)
            return "File watching command processed."
        
        # Create collapsible section for request processing
        processing_content = self.ui.info("Processing your request...")
        
        # Get relevant context
        relevant_context = self.context.get_relevant_context(request)
        if relevant_context:
            context_info = self.ui.info(f"Found {len(relevant_context)} relevant context items")
            processing_content += "\n" + context_info
        
        print(self.ui.section("Request Processing", processing_content, collapsible=True, expanded=False))
        
        try:
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt_with_context(request, relevant_context)
            
            # Send the enhanced prompt and get response from the configured API
            response = self._call_api(enhanced_prompt)
            
            # Extract plan and tool calls
            plan_text = response.get('content', '')
            tool_calls = response.get('tool_calls', [])
            
            # Create collapsible sections for different parts
            if plan_text:
                print(self.ui.section("Plan", plan_text, collapsible=True, expanded=True))
            
            # Execute tools if needed
            tools_used = []
            if tool_calls:
                execution_content = self.ui.subsection(f"Executing {len(tool_calls)} tool(s)")
                for i, tool_call in enumerate(tool_calls, 1):
                    tool_result = self._execute_tool(tool_call, i, len(tool_calls))
                    if tool_result:
                        tools_used.append(tool_result)
                        execution_content += f"\n  [{i}/{len(tool_calls)}] {tool_call['name']} - Completed"
                
                print(self.ui.section("Tool Execution", execution_content, collapsible=True, expanded=True))
                
                # Check if we need to continue with more tools
                self._check_for_more_work(request, plan_text)
                
                print(self.ui.success("Task completed successfully"))
                
                # Provide proactive suggestions
                self._provide_proactive_suggestions(request, tools_used)
            else:
                print(self.ui.info(plan_text if plan_text else "Request processed"))
            
            # Save interaction to context
            self.context.add_interaction(request, plan_text, tools_used)
                
        except Exception as e:
            print(self.ui.error(f"Error: {str(e)}"))
            print(self.ui.info("Please try again or rephrase your request"))

def main():
    """Main entry point for the CLI - works like Claude Code."""
    if len(sys.argv) < 2:
        # No arguments - enter Claude Code-style interactive mode
        print(colored("gcode", Colors.HIGHLIGHT + Colors.BOLD))
        print(colored("Your intelligent coding companion", Colors.SECONDARY))
        print()
        print(colored("What gcode does:", Colors.BOLD))
        print(colored("  • Write, analyze, and refactor code", Colors.INFO))
        print(colored("  • Generate tests and documentation", Colors.INFO))
        print(colored("  • Monitor code quality and security", Colors.INFO))
        print(colored("  • Manage git operations intelligently", Colors.INFO))
        print(colored("  • Provide real-time coding assistance", Colors.INFO))
        print()
        print(colored("Usage:", Colors.BOLD))
        print(colored("  gcode                    - Enter interactive mode (like Claude Code)", Colors.SUCCESS))
        print(colored("  gcode 'your request'     - Execute a single coding request", Colors.SUCCESS))
        print(colored("  gcode --help             - Show advanced options", Colors.SUCCESS))
        print()
        print(colored("Examples:", Colors.BOLD))
        print(colored("  gcode 'explain this function'", Colors.INFO))
        print(colored("  gcode 'refactor this code for better performance'", Colors.INFO))
        print(colored("  gcode 'generate unit tests for the auth module'", Colors.INFO))
        print(colored("  gcode 'commit my changes with a descriptive message'", Colors.INFO))
        print()
        print(colored("Type 'gcode --help' for advanced options and API management", Colors.HIGHLIGHT))
        print()
        
        # Enter Claude Code-style interactive mode
        agent = GeminiAgent()
        agent.converse("", interactive=True)
        return
    
    # Check for help command
    if sys.argv[1] == '--help':
        show_help()
        return
    
    # Check for interactive mode
    if sys.argv[1] == '--interactive':
        agent = GeminiAgent()
        agent.converse("", interactive=True)
    else:
        # Single request mode - like Claude Code
        prompt = " ".join(sys.argv[1:])
        agent = GeminiAgent()
        agent.converse(prompt, interactive=False)

def show_help():
    """Show comprehensive help and advanced options."""
    print(colored("gcode Help & Advanced Options", Colors.HIGHLIGHT + Colors.BOLD))
    print(colored("=" * 40, Colors.SECONDARY))
    print()
    
    print(colored("Basic Usage:", Colors.BOLD))
    print(colored("  gcode 'your request'           - Execute a single coding request", Colors.INFO))
    print(colored("  gcode --interactive            - Enter interactive mode", Colors.INFO))
    print(colored("  gcode --help                   - Show this help", Colors.INFO))
    print()
    
    print(colored("API Management:", Colors.BOLD))
    print(colored("  gcode --switch-api             - Change your AI service", Colors.INFO))
    print(colored("  gcode --show-config            - View current configuration", Colors.INFO))
    print(colored("  gcode --test-api               - Test your API connection", Colors.INFO))
    print()
    
    print(colored("Available AI Services:", Colors.BOLD))
    print(colored("  • Gemini (Google)              - Free tier, 50 requests/day", Colors.PRIMARY))
    print(colored("  • OpenAI (GPT-4o)              - Premium, unlimited requests", Colors.SUCCESS))
    print()
    
    print(colored("Advanced Features:", Colors.BOLD))
    print(colored("  • Real-time code monitoring    - Continuous quality checks", Colors.INFO))
    print(colored("  • Auto-fixing                  - Automatic code improvements", Colors.INFO))
    print(colored("  • Security scanning            - Vulnerability detection", Colors.INFO))
    print(colored("  • Performance profiling        - Code optimization insights", Colors.INFO))
    print(colored("  • Git integration              - Smart commit messages", Colors.INFO))
    print()
    
    print(colored("For more information, visit: https://github.com/your-repo/gcode", Colors.SECONDARY))

if __name__ == "__main__":
    main()

