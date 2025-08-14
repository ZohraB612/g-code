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
    performance_profiling,
    deep_codebase_analysis,
    analyze_code_quality
)
from dotenv import load_dotenv
import pickle
from .file_watcher import FileWatcher, create_file_watcher
from .analyzer import CodebaseAnalyzer, create_analyzer
from typing import List, Dict

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
    performance_profiling,
    deep_codebase_analysis,
    analyze_code_quality
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
        self.knowledge_graph_file = self.project_root / ".gcode_knowledge_graph.json"  # New
        self.conversation_history = []
        self.project_insights = {}
        self.knowledge_graph = {}  # New: deep codebase understanding
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
                print(f"Warning: Could not load context: {e}")
        
        # Load the knowledge graph for deep understanding
        if self.knowledge_graph_file.exists():
            try:
                with open(self.knowledge_graph_file, 'r') as f:
                    data = json.load(f)
                    # Remove metadata entries
                    self.knowledge_graph = {k: v for k, v in data.items() if not k.startswith('__')}
                    print(f"📚 Loaded knowledge graph: {len(self.knowledge_graph)} files")
            except Exception as e:
                print(f"Warning: Could not load knowledge graph: {e}")
    
    def save_context(self):
        """Save context to file."""
        data = {
            'conversation_history': self.conversation_history,
            'project_insights': self.project_insights,
            'last_analysis': self.last_analysis,
            'knowledge_graph_files': len(self.knowledge_graph)  # Track knowledge graph size
        }
        
        try:
            with open(self.context_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save context: {e}")
    
    def add_interaction(self, user_input: str, response: str, tools_used: List[str]):
        """Add a new interaction to the conversation history."""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'tools_used': tools_used
        }
        self.conversation_history.append(interaction)
        
        # Keep only last 50 interactions to prevent memory bloat
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
        
        self.save_context()
    
    def get_relevant_context(self, query: str) -> List[Dict]:
        """Get relevant context for a query using the knowledge graph."""
        relevant = []
        query_lower = query.lower()
        
        # Search through knowledge graph for relevant files
        for file_path, file_info in self.knowledge_graph.items():
            relevance_score = 0
            
            # Check if query mentions file types
            if any(ext in query_lower for ext in ['.py', '.js', '.java', '.cpp']):
                if file_info.get('file_type') in query_lower:
                    relevance_score += 5
            
            # Check if query mentions specific functions/classes
            if file_info.get('functions'):
                for func in file_info['functions']:
                    if func['name'].lower() in query_lower:
                        relevance_score += 3
            
            if file_info.get('classes'):
                for cls in file_info['classes']:
                    if cls['name'].lower() in query_lower:
                        relevance_score += 3
            
            # Check if query mentions frameworks
            if 'framework' in query_lower or 'django' in query_lower or 'flask' in query_lower:
                if file_info.get('file_type') == 'config':
                    relevance_score += 2
            
            # Check if query mentions architecture
            if any(word in query_lower for word in ['architecture', 'structure', 'dependencies']):
                if file_info.get('dependencies') or file_info.get('dependents'):
                    relevance_score += 2
            
            # Add to relevant if score is high enough
            if relevance_score > 0:
                relevant.append({
                    'file_path': file_path,
                    'relevance_score': relevance_score,
                    'file_info': file_info
                })
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:10]  # Return top 10 most relevant

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
        
        # Enhanced system prompt for autonomous end-to-end workflows
        self.system_prompt = """You are an expert software engineer and autonomous agent. Your primary goal is to achieve a user's objective by creating and executing a step-by-step plan.

When you receive a high-level goal, you must first decompose it into a series of smaller, actionable steps. Each step should involve one or more tool calls.

**Your response MUST be a JSON object containing a 'plan' key. The value should be an array of steps.**

Each step in the array should be an object with two keys:
1. `thought`: A brief explanation of what you are trying to accomplish in this step.
2. `tool_calls`: An array of tool calls required to complete this step.

Example:
User Goal: "Add a function to utils.py that adds two numbers, then test it."

Your JSON Response:
{
  "plan": [
    {
      "thought": "First, I need to read the existing content of utils.py to see where to add the new function.",
      "tool_calls": [
        {"name": "read_file", "arguments": {"path": "utils.py"}}
      ]
    },
    {
      "thought": "Now, I will append the new 'add' function to the file.",
      "tool_calls": [
        {"name": "write_file", "arguments": {"path": "utils.py", "content": "#... existing content ...\ndef add(a, b):\n    return a + b"}}
      ]
    },
    {
      "thought": "Next, I need to create a test file for this new function.",
      "tool_calls": [
        {"name": "create_test_file", "arguments": {"source_file": "utils.py"}}
      ]
    },
    {
      "thought": "Finally, I will run the tests to ensure everything is working correctly.",
      "tool_calls": [
        {"name": "run_tests", "arguments": {"test_directory": "tests/"}}
      ]
    }
  ]
}

**Key Principles:**
1. **Be Autonomous**: Complete the entire workflow without asking for permission
2. **Plan Thoroughly**: Think through all steps needed to achieve the goal
3. **Handle Errors**: If a step fails, analyze the error and create a recovery plan
4. **Use All Available Tools**: Leverage the full power of gcode's capabilities
5. **Provide Progress Updates**: Show what's happening at each step
6. **Self-Correct**: When things go wrong, automatically create a new plan

**Available Tools for Complex Workflows:**
- File operations: read_file, write_file, create_test_file
- Git operations: git_status, git_commit_with_ai_message, git_resolve_conflicts
- Code analysis: analyze_python_file, query_codebase, deep_codebase_analysis
- Testing: run_tests, generate_property_based_tests
- Quality: monitor_code_quality_continuous, auto_fix_common_issues
- Security: run_security_scan, performance_profiling

Think carefully and create a complete plan to achieve the user's entire goal."""
        
        # Initialize file watcher
        self.file_watcher = None
        self.watching_files = False
        self.auto_analysis = True
        self.auto_watch_files = True  # New: control auto-file-watching
        
        # Initialize workflow execution mode
        self.execution_mode = 'auto'  # Default to automatic execution
        
        # Initialize with project analysis
        self._analyze_project_context()
        
        # Generate project memory file (like Claude Code's CLAUDE.md)
        self.generate_project_memory()
        
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
                messages.append({"role": "assistant", "content": conv['response']})
            
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

    def _enhance_prompt_with_context(self, prompt: str, relevant_context: List[Dict]) -> str:
        """Enhance the user prompt with deep codebase context."""
        context_info = "\n\n--- Deep Codebase Context ---\n"
        
        # Add knowledge graph insights
        if self.context.knowledge_graph:
            context_info += "🧠 Deep Codebase Understanding:\n"
            
            # Add architecture overview
            arch = self.context.knowledge_graph.get('__architecture__', {})
            if arch:
                overview = arch.get('overview', {})
                context_info += f"🏗️  Architecture: {len(overview.get('main_modules', []))} main modules, "
                context_info += f"{len(overview.get('entry_points', []))} entry points\n"
                context_info += f"🧪 Test Coverage: {len(overview.get('test_files', []))} test files\n"
            
            # Add framework information
            patterns = self.context.knowledge_graph.get('__patterns__', {})
            if patterns:
                frameworks = patterns.get('patterns', {}).get('frameworks', [])
                if frameworks:
                    context_info += f"⚡ Frameworks: {', '.join(frameworks)}\n"
            
            # Add relevant file details
            if relevant_context:
                context_info += f"\n📁 Relevant Files ({len(relevant_context)}):\n"
                for item in relevant_context[:5]:  # Top 5 most relevant
                    file_info = item['file_info']
                    context_info += f"  • {item['file_path']}\n"
                    
                    # Add function/class info for Python files
                    if file_info.get('file_type') == 'python':
                        if file_info.get('functions'):
                            funcs = [f['name'] for f in file_info['functions'][:3]]  # Top 3 functions
                            context_info += f"    Functions: {', '.join(funcs)}\n"
                        if file_info.get('classes'):
                            classes = [c['name'] for c in file_info['classes'][:3]]  # Top 3 classes
                            context_info += f"    Classes: {', '.join(classes)}\n"
        
        # Add conversation history context
        if self.context.conversation_history:
            recent = self.context.conversation_history[-3:]  # Last 3 interactions
            context_info += f"\n💬 Recent Context ({len(recent)} interactions):\n"
            for conv in recent:
                context_info += f"  • {conv['user_input'][:50]}...\n"
        
        return prompt + context_info
    
    def query_codebase(self, question: str) -> str:
        """Query the knowledge graph with natural language questions."""
        if not self.context.knowledge_graph:
            return "❌ No knowledge graph available. Run project analysis first."
        
        # Try to use the analyzer's query method
        try:
            analyzer = create_analyzer()
            analyzer.knowledge_graph = self.context.knowledge_graph
            return analyzer.query(question)
        except Exception as e:
            return f"❌ Error querying codebase: {e}"

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
        """Analyze the project context with DEEP codebase understanding."""
        try:
            # Create collapsible section for project analysis
            analysis_content = self.ui.info("Analyzing project context...")
            
            # DEEP CODEBASE ANALYSIS - This is the key differentiator!
            print("🧠 Building deep codebase understanding...")
            analyzer = create_analyzer()
            knowledge_graph = analyzer.analyze()
            
            # Update context with knowledge graph
            self.context.knowledge_graph = knowledge_graph
            
            # Get traditional insights
            structure = get_project_structure()
            python_files = list(Path(".").rglob("*.py"))
            insights = {
                'total_python_files': len(python_files),
                'knowledge_graph_files': len(knowledge_graph),
                'has_deep_analysis': True
            }
            
            self.context.project_insights = insights
            self.context.last_analysis = datetime.now().isoformat()
            self.context.save_context()
            
            # Show deep analysis results
            arch = knowledge_graph.get('__architecture__', {})
            patterns = knowledge_graph.get('__patterns__', {})
            
            success_msg = self.ui.success(f"Deep analysis complete: {len(knowledge_graph)} files mapped")
            
            # Create the section with rich content
            analysis_summary = f"""
🧠 Deep Codebase Understanding Built:
📊 Files Analyzed: {len(knowledge_graph)}
🏗️  Architecture: {len(arch.get('overview', {}).get('main_modules', []))} main modules
🔗 Dependencies: Mapped across all files
⚡ Frameworks: {', '.join(patterns.get('patterns', {}).get('frameworks', ['None detected']))}
🧪 Testing: {len(arch.get('overview', {}).get('test_files', []))} test files
            """.strip()
            
            print(self.ui.section("Deep Project Analysis", 
                                 analysis_content + "\n" + analysis_summary + "\n" + success_msg, 
                                 collapsible=True, expanded=True))
            
        except Exception as e:
            print(self.ui.error(f"Could not perform deep analysis: {e}"))
            print("💡 Falling back to basic project analysis...")
            
            # Fallback to basic analysis
            try:
                structure = get_project_structure()
                python_files = list(Path(".").rglob("*.py"))
                insights = {
                    'total_python_files': len(python_files),
                    'has_deep_analysis': False
                }
                
                self.context.project_insights = insights
                self.context.last_analysis = datetime.now().isoformat()
                self.context.save_context()
                
                success_msg = self.ui.success(f"Basic analysis complete: {len(python_files)} Python files found")
                print(self.ui.section("Project Analysis", 
                                     analysis_content + "\n" + success_msg, 
                                     collapsible=True, expanded=True))
                
            except Exception as fallback_error:
                print(self.ui.error(f"Basic analysis also failed: {fallback_error}"))
    
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
                if result and not result.startswith("Error"):
                    print("✅ Auto-analysis complete!")
                    
                    # Extract key metrics from the analysis string
                    lines = []
                    issues = 0
                    
                    # Parse the result string to extract metrics
                    for line in result.split('\n'):
                        if 'Total lines:' in line:
                            lines.append(line.strip())
                        elif 'Long lines' in line or 'TODO items' in line or 'Print statements' in line:
                            issues += 1
                    
                    # Display summary
                    if lines:
                        print(f"📊 {lines[0]}")
                    if issues > 0:
                        print(f"🔍 Issues detected: {issues}")
                        print("💡 Use 'gcode fix-issues' to automatically fix detected problems.")
                else:
                    print("⚠️  Auto-analysis completed with warnings")
                    if result:
                        print(result[:200] + "..." if len(result) > 200 else result)
            
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
        
        print("\n🧠 Deep Codebase Understanding:")
        print("- 'query \"what functions are in agent.py?\"'")
        print("- 'query \"show me the architecture\"'")
        print("- 'query \"what frameworks are used?\"'")
        print("- 'query \"find files with authentication\"'")
        print("- 'query \"complexity analysis\"'")
        print("- 'query \"test coverage\"'")
        print("- 'generate' - Create GCODE.md project memory file")
        
        print("\n🚀 Autonomous Workflows:")
        print("- 'Create a new feature for user authentication'")
        print("- 'Build and test a new utility function'")
        print("- 'Set up automated testing for this project'")
        print("- 'Refactor the code for better performance'")
        print("- 'Implement CI/CD pipeline for deployment'")
        
        print("\n🎛️  Interactive Workflow Controls:")
        print("  • Full Auto: Execute everything automatically")
        print("  • Step-by-Step: Confirm each step before execution")
        print("  • Preview Only: Show what would happen without executing")
        print("  • Real-time status updates and progress tracking")
        print("  • To-do list view with step-by-step progress")
        print("  • User control over execution flow")
        
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
        """Process a user request with autonomous end-to-end workflow execution."""
        # Check for watch commands first
        if request.startswith('watch '):
            command = request[6:].strip()
            self.watch_commands(command)
            return "File watching command processed."
        
        # Check for codebase queries
        if request.startswith('query ') or request.startswith('ask '):
            question = request[6:] if request.startswith('query ') else request[4:]
            return self.query_codebase(question)
        
        # Check for deep analysis commands
        if request.startswith('analyze ') or request == 'analyze':
            if request == 'analyze':
                from .tools import deep_codebase_analysis
                return deep_codebase_analysis()
            else:
                # Extract file path if provided
                file_path = request[8:].strip()
                from .tools import analyze_code_quality
                return analyze_code_quality(file_path if file_path else None)
        
        # Check for project memory generation
        if request.startswith('generate ') or request == 'generate':
            if request == 'generate' or 'memory' in request.lower():
                self.generate_project_memory(force_update=True)
                return "Project memory generated successfully."
            else:
                return "Use 'generate' or 'generate memory' to create GCODE.md"
        
        # Check for help commands
        if request.lower() in ['help', '?']:
            self._show_deep_help()
            return "Help displayed."
        
        # Check if this is a complex workflow request
        if self._is_complex_workflow(request):
            print(self.ui.info("🎯 Detected complex workflow - switching to autonomous mode..."))
            return self._handle_complex_workflow(request)
        
        # Fall back to simple request processing for basic queries
        return self._handle_simple_request(request)
    
    def _is_complex_workflow(self, request: str) -> bool:
        """Determine if a request requires complex workflow execution."""
        workflow_keywords = [
            'create', 'build', 'implement', 'set up', 'configure', 'deploy',
            'add feature', 'refactor', 'migrate', 'update', 'fix', 'optimize',
            'test', 'validate', 'commit', 'push', 'pull request', 'workflow',
            'automate', 'integrate', 'deploy', 'release'
        ]
        
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in workflow_keywords)
    
    def _handle_complex_workflow(self, request: str):
        """Handle complex workflows using autonomous planning and execution."""
        print(self.ui.section("Complex Workflow", f"Processing: {request}", expanded=True))
        
        # Phase 1: Create the plan
        plan = self._create_plan(request)
        
        if not plan:
            print(self.ui.error("Failed to create a plan. Falling back to simple processing..."))
            return self._handle_simple_request(request)
        
        # Display the plan with interactive controls
        self._display_interactive_plan(plan, request)
        
        # Ask user for confirmation and control preferences
        if not self._get_workflow_confirmation(plan):
            print(self.ui.info("Workflow cancelled by user."))
            return "Workflow cancelled."
        
        # Phase 2: Execute the plan with user interaction
        success = self._execute_plan_interactive(plan, request)
        
        if success:
            print(self.ui.success("🎉 Complex workflow completed successfully!"))
            return "Workflow completed successfully."
        else:
            print(self.ui.error("❌ Complex workflow failed."))
            return "Workflow failed. Please check the errors above."
    
    def _display_interactive_plan(self, plan: list, request: str):
        """Display the execution plan with interactive to-do list view."""
        print(self.ui.section("📋 Execution Plan", f"Generated {len(plan)} steps to achieve: {request}", expanded=True))
        
        # Create interactive to-do list
        for i, step in enumerate(plan, 1):
            step_id = f"step_{i}"
            step_status = "⏳"  # Pending
            step_thought = step.get('thought', 'No description provided')
            tool_count = len(step.get('tool_calls', []))
            
            # Display step with status and details
            print(f"  {step_status} Step {i}: {step_thought}")
            print(f"     🔧 Tools: {tool_count} tool(s) required")
            
            # Show tool details if available
            tool_calls = step.get('tool_calls', [])
            if tool_calls:
                for j, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call.get('name', 'Unknown tool')
                    tool_args = tool_call.get('arguments', {})
                    print(f"        {j}. {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})")
            
            print()  # Spacing between steps
        
        # Show execution summary
        total_tools = sum(len(step.get('tool_calls', [])) for step in plan)
        print(f"📊 Execution Summary:")
        print(f"   • Total Steps: {len(plan)}")
        print(f"   • Total Tools: {total_tools}")
        print(f"   • Estimated Time: {self._estimate_execution_time(plan)}")
    
    def _estimate_execution_time(self, plan: list) -> str:
        """Estimate execution time based on plan complexity."""
        total_tools = sum(len(step.get('tool_calls', [])) for step in plan)
        
        if total_tools <= 3:
            return "1-2 minutes"
        elif total_tools <= 6:
            return "3-5 minutes"
        elif total_tools <= 10:
            return "5-10 minutes"
        else:
            return "10+ minutes"
    
    def _get_workflow_confirmation(self, plan: list) -> bool:
        """Get user confirmation and execution preferences."""
        print(self.ui.section("🎛️  Workflow Controls", "Configure execution preferences", expanded=True))
        
        print("Execution Options:")
        print("  1. 🚀 Full Auto - Execute everything automatically")
        print("  2. 🛑 Step-by-Step - Confirm each step before execution")
        print("  3. 🔍 Preview Only - Show what would happen without executing")
        print("  4. ❌ Cancel - Don't execute this workflow")
        
        while True:
            try:
                choice = input(f"{colored('Select option (1-4): ', Colors.PRIMARY)}").strip()
                
                if choice == '1':
                    self.execution_mode = 'auto'
                    print("✅ Full automatic execution enabled")
                    return True
                elif choice == '2':
                    self.execution_mode = 'step_by_step'
                    print("✅ Step-by-step execution enabled")
                    return True
                elif choice == '3':
                    self.execution_mode = 'preview'
                    print("✅ Preview mode enabled - no execution")
                    return True
                elif choice == '4':
                    print("❌ Workflow cancelled")
                    return False
                else:
                    print("⚠️  Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                print("\n❌ Workflow cancelled by user")
                return False
    
    def _execute_plan_interactive(self, plan: list, original_goal: str):
        """Execute the plan with interactive user control."""
        if self.execution_mode == 'preview':
            return self._preview_execution(plan, original_goal)
        
        print(self.ui.info(f"🚀 Executing plan with {len(plan)} steps..."))
        execution_history = []
        
        for i, step in enumerate(plan, 1):
            # Update step status in the to-do list
            self._update_step_status(i, "🔄", "In Progress")
            
            print(self.ui.section(f"Step {i}/{len(plan)}: {step['thought']}", expanded=True))
            
            # Get user confirmation for step-by-step mode
            if self.execution_mode == 'step_by_step':
                if not self._confirm_step_execution(i, step):
                    print(self.ui.warning(f"Step {i} skipped by user"))
                    self._update_step_status(i, "⏭️", "Skipped")
                    execution_history.append({
                        'step': step,
                        'results': [],
                        'status': 'Skipped'
                    })
                    continue
            
            tool_calls = step.get('tool_calls', [])
            if not tool_calls:
                print(self.ui.warning("No tool calls for this step - skipping"))
                self._update_step_status(i, "⏭️", "Skipped")
                execution_history.append({
                    'step': step,
                    'result': {'status': 'Skipped', 'reason': 'No tool calls'}, 
                    'status': 'Skipped'
                })
                continue
            
            step_successful = True
            step_results = []
            
            for j, tool_call in enumerate(tool_calls, 1):
                print(f"  🔧 Executing tool {j}/{len(tool_calls)}...")
                
                try:
                    # Execute the tool
                    tool_result = self._execute_tool(tool_call, j, len(tool_calls))
                    
                    if tool_result:
                        step_results.append(tool_result)
                            print(self.ui.error(f"Tool execution failed: {tool_result['result']}"))
                            step_successful = False
                            break
                        else:
                            print(f"  ✅ Tool {j} completed successfully")
                    else:
                        print(f"  ⚠️  Tool {j} returned no result")
                        step_results.append({'status': 'No result'})
                        
                except Exception as e:
                    print(self.ui.error(f"Tool {j} execution error: {e}")
                    step_results.append({'status': 'Error', 'error': str(e)})
                    step_successful = False
                    break
            
            # Update step status based on result
            if step_successful:
                self._update_step_status(i, "✅", "Completed")
            else:
                self._update_step_status(i, "❌", "Failed")
            
            # Record step execution
            execution_history.append({
                'step': step,
                'results': step_results,
                'status': 'Success' if step_successful else 'Failed'
            })
            
            if not step_successful:
                print(self.ui.error(f"❌ Step {i} failed. Attempting to self-correct..."))
                
                # Try self-correction
                new_plan = self._self_correct(original_goal, execution_history)
                if new_plan:
                    print(self.ui.info("🔄 Self-correction successful! Executing new plan..."))
                    return self._execute_plan_interactive(new_plan, original_goal)
                else:
                    print(self.ui.error("❌ Self-correction failed. Aborting workflow."))
                    return False
        
        print(self.ui.success("✅ Workflow completed successfully!"))
        return True
    
    def _preview_execution(self, plan: list, original_goal: str):
        """Preview what would happen during execution without actually running it."""
        print(self.ui.section("🔍 Execution Preview", "Showing what would happen during execution", expanded=True))
        
        for i, step in enumerate(plan, 1):
            print(f"📋 Step {i}: {step['thought']}")
            
            tool_calls = step.get('tool_calls', [])
            if tool_calls:
                print(f"   🔧 Would execute {len(tool_calls)} tool(s):")
                for j, tool_call in enumerate(tool_calls, 1):
                    tool_name = tool_call.get('name', 'Unknown tool')
                    tool_args = tool_call.get('arguments', {})
                    print(f"      {j}. {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})")
            else:
                print("   ⚠️  No tools to execute")
            print()
        
        print("💡 This is a preview only. No actual changes were made.")
        print("💡 Use option 1 (Full Auto) or 2 (Step-by-Step) to actually execute.")
        return True
    
    def _confirm_step_execution(self, step_num: int, step: dict) -> bool:
        """Get user confirmation for step execution in step-by-step mode."""
        print(f"\n⏸️  Step {step_num} ready for execution:")
        print(f"   {step['thought']}")
        
        while True:
            try:
                choice = input(f"{colored('Execute this step? (y/n/skip): ", Colors.PRIMARY)}").strip().lower()
                
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                elif choice in ['s', 'skip']:
                    return False
                else:
                    print("⚠️  Please enter 'y' (yes), 'n' (no), or 's' (skip)")
            except KeyboardInterrupt:
                print("\n❌ Step execution cancelled")
                return False
    
    def _update_step_status(self, step_num: int, status_icon: str, status_text: str):
        """Update the visual status of a step in the to-do list."""
        # This would update the display in real-time
        # For now, we'll just print the status update
        print(f"  📊 Step {step_num} status: {status_icon} {status_text}")
    
    def _handle_simple_request(self, request: str):
        """Handle simple requests using the traditional approach."""
        # Create collapsible section for request processing
        processing_content = self.ui.info("Processing your request...")
        
        # Get relevant context using deep understanding
        relevant_context = self.context.get_relevant_context(request)
        if relevant_context:
            context_info = self.ui.info(f"Found {len(relevant_context)} relevant context items")
            print(self.ui.section("Relevant Context", context_info, collapsible=True, expanded=False))
        
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
        
        return "Request processed successfully"
    
    def _show_deep_help(self):
        """Show comprehensive help for deep codebase understanding features."""
        print("\n🧠 Deep Codebase Understanding - Help")
        print("=" * 50)
        
        print("\n📚 Knowledge Graph Commands:")
        print("  query <question>     - Ask questions about your codebase")
        print("  ask <question>        - Alternative to query")
        print("  analyze               - Perform deep codebase analysis")
        print("  analyze <file>        - Analyze specific file quality")
        print("  generate              - Create GCODE.md project memory file")
        
        print("\n🔍 Sample Questions You Can Ask:")
        print("  • 'query what functions are in agent.py?'")
        print("  • 'query show me the architecture'")
        print("  • 'query what frameworks are used?'")
        print("  • 'query find files with authentication'")
        print("  • 'query complexity analysis'")
        print("  • 'query test coverage'")
        print("  • 'query what are the dependencies of tools.py?'")
        
        print("\n💡 What This Gives You:")
        print("  • Semantic understanding of your entire codebase")
        print("  • Automatic dependency mapping")
        print("  • Framework and technology detection")
        print("  • Code complexity analysis")
        print("  • Test coverage insights")
        print("  • Architecture overview")
        
        print("\n🚀 This is what makes Claude Code so powerful!")
        print("   No more manually selecting files - gcode understands your project!")
        
        print("\n🎯 Autonomous Workflow Capabilities:")
        print("  • Complex task decomposition and planning")
        print("  • Multi-step execution with progress tracking")
        print("  • Automatic error handling and self-correction")
        print("  • End-to-end workflow completion")
        print("  • No intermediate user prompts needed")
        
        print("\n💡 Example Complex Workflows:")
        print("  • 'Create a new authentication system'")
        print("  • 'Build and deploy a web API'")
        print("  • 'Set up comprehensive testing suite'")
        print("  • 'Refactor legacy code for modern standards'")
        print("  • 'Implement CI/CD pipeline with GitHub Actions'")
        
        print("\n🎛️  Interactive Workflow Controls:")
        print("  • Full Auto: Execute everything automatically")
        print("  • Step-by-Step: Confirm each step before execution")
        print("  • Preview Only: Show what would happen without executing")
        print("  • Real-time status updates and progress tracking")
        print("  • To-do list view with step-by-step progress")
        print("  • User control over execution flow")

    def generate_project_memory(self, force_update: bool = False):
        """Generate a GCODE.md file with project memory and knowledge graph summary.
        This mimics Claude Code's CLAUDE.md functionality for persistent project context."""
        
        gcode_md_path = Path("GCODE.md")
        
        # Check if we should update
        if gcode_md_path.exists() and not force_update:
            last_modified = datetime.fromtimestamp(gcode_md_path.stat().st_mtime)
            if (datetime.now() - last_modified).days < 1:  # Update if older than 1 day
                print("📚 GCODE.md is up to date (less than 1 day old)")
                return
        
        print("🧠 Generating project memory file (GCODE.md)...")
        
        try:
            # Ensure we have the latest knowledge graph
            if not self.context.knowledge_graph:
                print("📊 Building knowledge graph first...")
                self._analyze_project_context()
            
            # Extract key information
            knowledge_graph = self.context.knowledge_graph
            arch = knowledge_graph.get('__architecture__', {})
            patterns = knowledge_graph.get('__patterns__', {})
            
            # Generate the markdown content
            content = f"""# GCODE Project Memory

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 🏗️ Project Overview

This is a **gcode** project - an intelligent coding companion with deep codebase understanding capabilities.

### 📊 Project Statistics
- **Total Files**: {len(knowledge_graph)}
- **Python Files**: {len([f for f in knowledge_graph.values() if f.get('file_type') == 'python'])}
- **Main Modules**: {len(arch.get('overview', {}).get('main_modules', []))}
- **Test Files**: {len(arch.get('overview', {}).get('test_files', []))}
- **Test Coverage**: {arch.get('test_coverage', 0):.1%}

## 🏛️ Architecture

### Entry Points
{chr(10).join([f"- `{ep}`" for ep in arch.get('overview', {}).get('entry_points', [])]) or "- None detected"}

### Main Modules
{chr(10).join([f"- `{mod}`" for mod in arch.get('overview', {}).get('main_modules', [])[:10]]) or "- None detected"}

### Packages
{chr(10).join([f"- `{pkg}`" for pkg in arch.get('overview', {}).get('packages', [])]) or "- None detected"}

## ⚡ Technologies

### Frameworks
{chr(10).join([f"- {fw}" for fw in patterns.get('patterns', {}).get('frameworks', [])]) or "- None detected"}

### Testing
{chr(10).join([f"- {test}" for test in patterns.get('patterns', {}).get('testing_frameworks', [])]) or "- None detected"}

### Build Tools
{chr(10).join([f"- {tool}" for tool in patterns.get('patterns', {}).get('build_tools', [])]) or "- None detected"}

## 🔍 Key Files Analysis

### High-Level Structure
"""
            
            # Add key files with their purposes
            key_files = []
            for file_path, file_info in knowledge_graph.items():
                if file_info.get('file_type') == 'python':
                    purpose = self._determine_file_purpose(file_path, file_info)
                    key_files.append((file_path, purpose))
            
            # Sort by importance
            key_files.sort(key=lambda x: self._calculate_file_importance(x[0], x[1]), reverse=True)
            
            for file_path, purpose in key_files[:15]:  # Top 15 most important
                content += f"- **`{file_path}`**: {purpose}\n"
            
            content += f"""

### Dependencies Map
"""
            
            # Show dependency relationships
            dependency_map = {}
            for file_path, file_info in knowledge_graph.items():
                if file_info.get('dependencies'):
                    dependency_map[file_path] = file_info['dependencies']
            
            if dependency_map:
                for file_path, deps in list(dependency_map.items())[:10]:  # Top 10
                    content += f"- **`{file_path}`** depends on: {', '.join([f'`{dep}`' for dep in deps[:3]])}\n"
            else:
                content += "- No complex dependencies detected\n"
            
            content += f"""

## 🧪 Testing & Quality

### Test Coverage
- **Test Files**: {len(arch.get('overview', {}).get('test_files', []))}
- **Coverage Ratio**: {arch.get('test_coverage', 0):.1%}
- **Status**: {'Excellent' if arch.get('test_coverage', 0) >= 0.8 else 'Good' if arch.get('test_coverage', 0) >= 0.5 else 'Needs improvement'}

### Code Quality
"""
            
            # Add complexity analysis
            complex_files = []
            for file_path, file_info in knowledge_graph.items():
                if file_info.get('file_type') == 'python' and file_info.get('complexity', 0) > 10:
                    complex_files.append((file_path, file_info['complexity']))
            
            if complex_files:
                complex_files.sort(key=lambda x: x[1], reverse=True)
                content += "**High Complexity Files (>10):**\n"
                for file_path, complexity in complex_files[:5]:
                    content += f"- `{file_path}`: complexity {complexity}\n"
            else:
                content += "- All files have reasonable complexity\n"
            
            content += f"""

## 💡 Recent Context

### Last {min(5, len(self.context.conversation_history))} Interactions
"""
            
            # Add recent conversation context
            for conv in self.context.conversation_history[-5:]:
                content += f"- **{conv['user_input'][:60]}...**\n"
            
            content += f"""

## 🚀 How to Use This Project

### With gcode CLI
```bash
# Ask questions about the codebase
gcode 'query what functions are in agent.py?'
gcode 'query show me the architecture'
gcode 'query what frameworks are used?'

# Analyze code quality
gcode 'analyze'
gcode 'analyze gcode/agent.py'

# Start file watching
gcode 'watch start'
```

### Interactive Mode
```bash
gcode
> query find files with test
> query complexity analysis
> query test coverage
```

## 🔄 Auto-Update

This file is automatically updated when:
- You run `gcode 'analyze'`
- You start a new gcode session
- The project structure changes significantly

---

*Generated by gcode - Your intelligent coding companion* 🚀
"""
            
            # Write the file
            with open(gcode_md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Project memory saved to: {gcode_md_path}")
            print("📚 This file provides persistent context for future gcode sessions")
            
        except Exception as e:
            print(f"❌ Error generating project memory: {e}")
    
    def _determine_file_purpose(self, file_path: str, file_info: dict) -> str:
        """Determine the purpose of a file based on its content and structure."""
        path_lower = file_path.lower()
        
        # Check for common patterns
        if 'test' in path_lower:
            return "Test file"
        elif 'agent' in path_lower:
            return "Main agent logic and AI integration"
        elif 'tools' in path_lower:
            return "Utility functions and tools"
        elif 'analyzer' in path_lower:
            return "Code analysis and knowledge graph building"
        elif 'watcher' in path_lower:
            return "File change monitoring and auto-analysis"
        elif 'cli' in path_lower:
            return "Command-line interface"
        elif 'setup' in path_lower:
            return "Package configuration and installation"
        elif 'requirements' in path_lower:
            return "Python dependencies"
        elif 'docker' in path_lower:
            return "Container configuration"
        elif 'readme' in path_lower:
            return "Project documentation"
        elif 'config' in path_lower:
            return "Configuration and settings"
        else:
            # Analyze content for clues
            functions = file_info.get('functions', [])
            classes = file_info.get('classes', [])
            
            if functions and classes:
                return f"Module with {len(functions)} functions and {len(classes)} classes"
            elif functions:
                return f"Utility module with {len(functions)} functions"
            elif classes:
                return f"Class-based module with {len(classes)} classes"
            else:
                return "Configuration or data file"
    
    def _calculate_file_importance(self, file_path: str, purpose: str) -> int:
        """Calculate the importance score of a file."""
        score = 0
        
        # Core functionality files get higher scores
        if 'agent' in file_path.lower():
            score += 100
        elif 'tools' in file_path.lower():
            score += 80
        elif 'analyzer' in file_path.lower():
            score += 70
        elif 'watcher' in file_path.lower():
            score += 60
        
        # Test files get lower scores
        if 'test' in file_path.lower():
            score -= 50
        
        # Configuration files get medium scores
        if any(x in file_path.lower() for x in ['setup', 'requirements', 'docker']):
            score += 30
        
        return score

    def _create_plan(self, goal: str):
        """
        Calls the LLM to create a step-by-step plan to achieve a goal.
        This is the core of the autonomous workflow system.
        """
        print(self.ui.info("🤔 Thinking and creating a plan..."))
        
        try:
            # Use the enhanced system prompt to generate a structured plan
            response = self._call_api(goal)
            
            # Try to extract JSON plan from the response
            content = response.get('content', '')
            
            # Look for JSON in the response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*"plan".*\}', content, re.DOTALL)
            if json_match:
                try:
                    plan_json = json.loads(json_match.group())
                    plan = plan_json.get('plan', [])
                    if plan:
                        print(f"✅ Generated plan with {len(plan)} steps")
                        return plan
                except json.JSONDecodeError:
                    pass
            
            # If no JSON found, try to parse the entire response
            try:
                plan_json = json.loads(content)
                plan = plan_json.get('plan', [])
                if plan:
                    print(f"✅ Generated plan with {len(plan)} steps")
                    return plan
            except json.JSONDecodeError:
                pass
            
            # Fallback: try to extract plan from text
            print("⚠️  No structured plan found, attempting to extract from text...")
            return self._extract_plan_from_text(content, goal)
            
        except Exception as e:
            print(self.ui.error(f"Could not generate a valid plan: {e}"))
            print(self.ui.warning("The model did not return a valid JSON plan. Please try rephrasing your request."))
            return None
    
    def _extract_plan_from_text(self, text: str, goal: str):
        """Fallback method to extract a plan from unstructured text."""
        try:
            # Look for numbered steps or bullet points
            lines = text.split('\n')
            plan = []
            current_step = None
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line) or line.startswith('-') or line.startswith('•'):
                    # This looks like a step
                    if current_step:
                        plan.append(current_step)
                    
                    # Extract the thought from the line
                    thought = re.sub(r'^\d+\.\s*', '', line)
                    thought = re.sub(r'^[-•]\s*', '', thought)
                    
                    current_step = {
                        'thought': thought,
                        'tool_calls': []
                    }
                elif current_step and line:
                    # This might be additional context for the current step
                    current_step['thought'] += ' ' + line
            
            # Add the last step
            if current_step:
                plan.append(current_step)
            
            if plan:
                print(f"✅ Extracted {len(plan)} steps from text")
                return plan
            
        except Exception as e:
            print(f"⚠️  Failed to extract plan from text: {e}")
        
        return None

    def _execute_plan(self, plan: list, original_goal: str):
        """
        Executes a plan step-by-step, with self-correction capabilities.
        This is the core execution engine for autonomous workflows.
        """
        print(self.ui.info(f"🚀 Executing plan with {len(plan)} steps..."))
        execution_history = []
        
        for i, step in enumerate(plan, 1):
            print(self.ui.section(f"Step {i}/{len(plan)}: {step['thought']}", expanded=True))
            
            tool_calls = step.get('tool_calls', [])
            if not tool_calls:
                print(self.ui.warning("No tool calls for this step - skipping"))
                execution_history.append({
                    'step': step, 
                    'result': {'status': 'Skipped', 'reason': 'No tool calls'}, 
                    'status': 'Skipped'
                })
                continue
            
            step_successful = True
            step_results = []
            
            for j, tool_call in enumerate(tool_calls, 1):
                print(f"  🔧 Executing tool {j}/{len(tool_calls)}...")
                
                try:
                    # Execute the tool
                    tool_result = self._execute_tool(tool_call, j, len(tool_calls))
                    
                    if tool_result:
                        step_results.append(tool_result)
                        
                        # Check if the tool execution failed
                        if isinstance(tool_result, dict) and "Error:" in str(tool_result.get('result', '')):
                            print(self.ui.error(f"Tool execution failed: {tool_result['result']}"))
                            step_successful = False
                            break
                        else:
                            print(f"  ✅ Tool {j} completed successfully")
                    else:
                        print(f"  ⚠️  Tool {j} returned no result")
                        step_results.append({'status': 'No result'})
                        
                except Exception as e:
                    print(self.ui.error(f"Tool {j} execution error: {e}"))
                    step_results.append({'status': 'Error', 'error': str(e)})
                    step_successful = False
                    break
            
            # Record step execution
            execution_history.append({
                'step': step,
                'results': step_results,
                'status': 'Success' if step_successful else 'Failed'
            })
            
            if not step_successful:
                print(self.ui.error(f"❌ Step {i} failed. Attempting to self-correct..."))
                
                # Try self-correction
                new_plan = self._self_correct(original_goal, execution_history)
                if new_plan:
                    print(self.ui.info("🔄 Self-correction successful! Executing new plan..."))
                    return self._execute_plan(new_plan, original_goal)
                else:
                    print(self.ui.error("❌ Self-correction failed. Aborting workflow."))
                    return False
        
        print(self.ui.success("✅ Workflow completed successfully!"))
        return True
    
    def _self_correct(self, original_goal: str, history: list):
        """
        Analyzes a failed execution and creates a new plan to recover.
        This enables the autonomous loop for error handling.
        """
        print(self.ui.info("🤖 Analyzing failure and creating a recovery plan..."))
        
        # Create a correction prompt
        correction_prompt = f"""An autonomous agent was trying to achieve the following goal: '{original_goal}'

It created a plan and failed. Here is the execution history:
{json.dumps(history, indent=2)}

The last step failed. Analyze the error and create a new, corrected plan to achieve the original goal.

**Your response MUST be a JSON object with a 'plan' key, just like before.**

Focus on:
1. What went wrong in the failed step
2. How to fix or work around the issue
3. Alternative approaches to achieve the goal
4. Any additional steps needed for recovery

Be specific about the fixes and ensure the new plan addresses the root cause of the failure."""
        
        try:
            # Call the API with the correction prompt
            response = self._call_api(correction_prompt)
            content = response.get('content', '')
            
            # Try to extract the new plan
            import json
            import re
            
            json_match = re.search(r'\{.*"plan".*\}', content, re.DOTALL)
            if json_match:
                try:
                    new_plan_json = json.loads(json_match.group())
                    new_plan = new_plan_json.get('plan', [])
                    if new_plan:
                        print(f"✅ Generated recovery plan with {len(new_plan)} steps")
                        return new_plan
                except json.JSONDecodeError:
                    pass
            
            # Fallback to text extraction
            return self._extract_plan_from_text(content, original_goal)
            
        except Exception as e:
            print(self.ui.error(f"Failed to generate a correction plan: {e}"))
            return None

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

