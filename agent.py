import google.generativeai as genai
import os
import sys
import json
import time
import threading
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

class ProfessionalUI:
    """Professional UI system with collapsible sections and clean formatting."""
    
    def __init__(self):
        self.section_id = 0
        self.active_sections = {}
    
    def section(self, title, collapsible=True, expanded=False):
        """Create a collapsible section with professional styling."""
        self.section_id += 1
        section_id = self.section_id
        
        if collapsible:
            status = "▼" if expanded else "▶"
            header = f"{status} {title}"
            if expanded:
                self.active_sections[section_id] = True
                return f"\n{colored(header, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}"
            else:
                return f"\n{colored(header, Colors.PRIMARY)}\n{colored('─' * len(title), Colors.SECONDARY)}"
        else:
            return f"\n{colored(title, Colors.PRIMARY, bold=True)}\n{colored('─' * len(title), Colors.SECONDARY)}"
    
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

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    """A conversational agent powered by the Gemini model with advanced capabilities."""
    
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        """Initializes the agent with a model and advanced context management."""
        self.model = genai.GenerativeModel(model_name=model_name, tools=tools)
        self.context = ProjectContext()
        self.ui = ProfessionalUI()
        
        # Enhanced system prompt for advanced capabilities
        self.chat = self.model.start_chat(history=[
            {"role": "user", "parts": [
                "You are an expert software engineer and pair-programmer, similar to Cursor or Claude Code. "
                "Your goal is to help with coding projects efficiently and conversationally. "
                "You have access to tools to read/write files, run commands, and analyze project structure. "
                "Key principles:\n"
                "1. Be proactive and autonomous - use tools when needed without asking permission\n"
                "2. Provide clear, actionable plans before executing\n"
                "3. Think step-by-step and explain your reasoning\n"
                "4. Be conversational and helpful, not robotic\n"
                "5. When you need to use tools, explain what you're doing and why\n"
                "6. Always provide a summary of what was accomplished\n"
                "7. IMPORTANT: Provide ALL necessary tool calls in a single response to complete the entire task\n"
                "8. Don't stop after one tool - think through the complete workflow and provide all tools needed\n"
                "9. Analyze the project context and suggest improvements proactively\n"
                "10. Remember previous work and build upon it intelligently\n"
                "11. Handle edge cases and errors gracefully\n"
                "12. Suggest next steps and improvements after completing tasks\n"
                "Remember: You're a coding partner, not just a tool executor. Complete the entire task in one go and be proactive about helping improve the codebase."
            ]}
        ])
        
        # Initialize with project analysis
        self._analyze_project_context()
    
    def _analyze_project_context(self):
        """Analyze the project context to understand the codebase."""
        try:
            print(self.ui.section("Project Analysis", collapsible=False))
            print(self.ui.info("Analyzing project context..."))
            
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
            
            print(self.ui.success(f"Project analyzed: {insights['total_python_files']} Python files found"))
            
        except Exception as e:
            print(self.ui.error(f"Could not analyze project: {e}"))
    
    def converse(self, prompt: str, interactive=False):
        """Handles the conversation flow with advanced context awareness."""
        if interactive:
            print(self.ui.section("Interactive Mode", collapsible=False))
            print(self.ui.info("Type 'exit' or 'quit' to end the session"))
            print(self.ui.info("Type 'help' for available commands"))
            print(self.ui.info("Type 'context' to see project insights"))
            print(colored("─" * 50, Colors.SECONDARY))
            
            while True:
                try:
                    user_input = input(colored("\nYou: ", Colors.PRIMARY)).strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        print(self.ui.info("Session ended. Goodbye!"))
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'context':
                        self._show_project_context()
                        continue
                    elif not user_input:
                        continue
                    
                    self._process_request(user_input)
                    
                except KeyboardInterrupt:
                    print(self.ui.info("Session interrupted. Goodbye!"))
                    break
                except EOFError:
                    print(self.ui.info("End of input. Goodbye!"))
                    break
        else:
            # Single request mode
            self._process_request(prompt)

    def _process_request(self, prompt: str):
        """Process a single user request with enhanced context awareness."""
        print(self.ui.section("Request Processing", collapsible=False))
        print(self.ui.info("Processing your request..."))
        
        # Get relevant context
        relevant_context = self.context.get_relevant_context(prompt)
        if relevant_context:
            print(self.ui.info(f"Found {len(relevant_context)} relevant context items"))
        
        try:
            # Enhance prompt with context
            enhanced_prompt = self._enhance_prompt_with_context(prompt, relevant_context)
            
            # Send the enhanced prompt and get response
            response = self.chat.send_message(enhanced_prompt)
            
            # Extract plan and tool calls
            response_parts = response.candidates[0].content.parts
            plan_text = "".join(part.text for part in response_parts if part.text).strip()
            tool_calls = [part.function_call for part in response_parts if part.function_call]
            
            # Display the plan
            if plan_text:
                print(self.ui.subsection("Plan"))
                print(plan_text)
            
            # Execute tools if needed
            tools_used = []
            if tool_calls:
                print(self.ui.subsection(f"Executing {len(tool_calls)} tool(s)"))
                for i, tool_call in enumerate(tool_calls, 1):
                    tool_result = self._execute_tool(tool_call, i, len(tool_calls))
                    if tool_result:
                        tools_used.append(tool_result)
                
                # Check if we need to continue with more tools
                self._check_for_more_work(prompt, plan_text)
                
                print(self.ui.success("Task completed successfully"))
                
                # Provide proactive suggestions
                self._provide_proactive_suggestions(prompt, tools_used)
            else:
                print(self.ui.info(plan_text if plan_text else "Request processed"))
            
            # Save interaction to context
            self.context.add_interaction(prompt, plan_text, tools_used)
                
        except Exception as e:
            print(self.ui.error(f"Error: {str(e)}"))
            print(self.ui.info("Please try again or rephrase your request"))

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
            
            response = self.chat.send_message(suggestions_prompt)
            suggestions = "".join(part.text for part in response.candidates[0].content.parts if part.text).strip()
            
            if suggestions:
                print(self.ui.subsection("Proactive Suggestions"))
                print(suggestions)
                
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
                response = self.chat.send_message(follow_up_prompt)
                
                response_parts = response.candidates[0].content.parts
                additional_tool_calls = [part.function_call for part in response_parts if part.function_call]
                
                if additional_tool_calls:
                    print(self.ui.subsection(f"Executing {len(additional_tool_calls)} additional tool(s)"))
                    for i, tool_call in enumerate(additional_tool_calls, 1):
                        self._execute_tool(tool_call, i, len(additional_tool_calls))
                    
                    print(self.ui.success("Additional work completed"))
                
            except Exception as e:
                print(self.ui.warning(f"Could not check for additional work: {e}"))

    def _execute_tool(self, tool_call, current, total):
        """Execute a single tool call with professional feedback."""
        func_name = tool_call.name
        func_to_call = AVAILABLE_TOOLS.get(func_name)
        
        if not func_to_call:
            print(self.ui.error(f"Unknown tool '{func_name}'"))
            return None
        
        func_args = dict(tool_call.args)
        
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
        
        print(self.ui.section("Project Context & Insights", collapsible=False))
        
        if insights:
            print(f"Python Files: {insights.get('total_python_files', 0)}")
            if insights.get('main_files'):
                print(f"Main Files: {', '.join(insights['main_files'])}")
            
            print(f"Requirements: {'Yes' if insights.get('has_requirements') else 'No'}")
            print(f"Setup Files: {'Yes' if insights.get('has_setup') else 'No'}")
            
            if insights.get('last_analysis'):
                print(f"Last Analyzed: {insights['last_analysis'][:19]}")
        else:
            print("No project insights available yet.")
        
        # Show recent conversation history
        if self.context.conversation_history:
            print(f"\nRecent Conversations: {len(self.context.conversation_history)}")
            for i, conv in enumerate(self.context.conversation_history[-3:], 1):
                print(f"   {i}. {conv['user_input'][:50]}...")

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print(colored("Gemini Agent CLI", Colors.PRIMARY + Colors.BOLD))
        print(colored("A Cursor/Claude Code-level coding assistant", Colors.SECONDARY))
        print(colored("\nUsage:", Colors.BOLD))
        print(colored("  python -m gcode.cli 'your request here'", Colors.SUCCESS))
        print(colored("  python -m gcode.cli --interactive", Colors.SUCCESS))
        print(colored("\nExamples:", Colors.BOLD))
        print(colored("  python -m gcode.cli 'create a hello world file'", Colors.INFO))
        print(colored("  python -m gcode.cli 'analyze this code and suggest improvements'", Colors.INFO))
        print(colored("  python -m gcode.cli --interactive", Colors.INFO))
        return
    
    # Check for interactive mode
    if sys.argv[1] == '--interactive':
        agent = GeminiAgent()
        agent.converse("", interactive=True)
    else:
        # Single request mode
        prompt = " ".join(sys.argv[1:])
        agent = GeminiAgent()
        agent.converse(prompt, interactive=False)

if __name__ == "__main__":
    main()

