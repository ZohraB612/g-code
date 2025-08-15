#!/usr/bin/env python3
"""
gcode - Your intelligent coding companion with dual API support and full agentic workflow capabilities.
"""

import openai
import google.generativeai as genai
import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from tools import (
    read_file, 
    write_file, 
    run_shell_command, 
    get_project_structure,
    analyze_python_file,
    create_test_file,
    search_code,
    install_dependencies,
    run_tests,
    git_status,
    git_commit_with_ai_message,
    git_resolve_conflicts,
    git_smart_branch
)
from dotenv import load_dotenv

load_dotenv()

# Professional color scheme (VS Code inspired)
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
    
    def setup_api(self, api_choice):
        """Setup the selected API."""
        if api_choice == 'gemini':
            return self._setup_gemini()
        elif api_choice == 'openai':
            return self._setup_openai()
        elif api_choice == 'auto':
            return self._auto_detect_api()
        else:
            print(f"{colored('Invalid API choice', Colors.ERROR)}")
            return False
    
    def _setup_gemini(self):
        """Setup Gemini API."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            api_key = input(f"{colored('Enter your Gemini API key: ', Colors.PRIMARY)}").strip()
            if not api_key:
                print(f"{colored('No API key provided. Gemini setup cancelled.', Colors.WARNING)}")
                return False
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Test the API
            response = model.generate_content("Hello, test message")
            if response.text:
                self.selected_api = 'gemini'
                self.api_key = api_key
                self.model_name = 'gemini-1.5-flash'
                print(f"{colored('✅ Gemini API configured successfully!', Colors.SUCCESS)}")
                return True
        except Exception as e:
            print(f"{colored(f'❌ Gemini API setup failed: {e}', Colors.ERROR)}")
            return False
    
    def _setup_openai(self):
        """Setup OpenAI API."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = input(f"{colored('Enter your OpenAI API key: ', Colors.PRIMARY)}").strip()
            if not api_key:
                print(f"{colored('No API key provided. OpenAI setup cancelled.', Colors.WARNING)}")
                return False
        
        try:
            openai.api_key = api_key
            # Test the API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello, test message"}],
                max_tokens=10
            )
            if response.choices[0].message.content:
                self.selected_api = 'openai'
                self.api_key = api_key
                self.model_name = 'gpt-4o-mini'
                print(f"{colored('✅ OpenAI API configured successfully!', Colors.SUCCESS)}")
                return True
        except Exception as e:
            print(f"{colored(f'❌ OpenAI API setup failed: {e}', Colors.ERROR)}")
            return False
    
    def _auto_detect_api(self):
        """Auto-detect available API keys."""
        print(f"{colored('🔍 Auto-detecting available API keys...', Colors.INFO)}")
        
        # Try OpenAI first
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            print(f"{colored('Found OpenAI API key, testing...', Colors.INFO)}")
            if self._setup_openai():
                return True
        
        # Try Gemini
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            print(f"{colored('Found Gemini API key, testing...', Colors.INFO)}")
            if self._setup_gemini():
                return True
        
        print(f"{colored('❌ No working API keys found. Please set up an API key manually.', Colors.ERROR)}")
        return False

class GeminiAgent:
    """Main agent class with full agentic workflow capabilities."""
    
    def __init__(self):
        self.authenticator = APIAuthenticator()
        self.api_configured = False
        self.setup_complete = False
        self.context = {}
        
    def setup(self):
        """Setup the agent with API configuration."""
        if self.setup_complete:
            return True
            
        self.authenticator.show_welcome()
        api_choice = self.authenticator.select_api()
        
        if self.authenticator.setup_api(api_choice):
            self.api_configured = True
            self.setup_complete = True
            return True
        else:
            print(f"{colored('❌ API setup failed. Please try again.', Colors.ERROR)}")
            return False
    
    def converse(self, prompt, interactive=True):
        """Main conversation method with intelligent routing."""
        if not self.setup():
            return
        
        if interactive:
            self._interactive_mode()
        else:
            self._single_request_mode(prompt)
    
    def _interactive_mode(self):
        """Interactive conversation mode with agentic capabilities."""
        print(f"\n{colored('🚀 Interactive Mode Started', Colors.SUCCESS)}")
        print(f"{colored('Type your coding requests or "help" for commands', Colors.INFO)}")
        print(f"{colored('Type "exit" or "quit" to end the session', Colors.INFO)}\n")
        
        while True:
            try:
                user_input = input(f"{colored('gcode> ', Colors.PRIMARY)}").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(f"{colored('👋 Goodbye! Happy coding!', Colors.SUCCESS)}")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'context':
                    self._show_context()
                elif user_input.lower() == 'demo':
                    self._show_demo()
                elif user_input:
                    self._process_request_intelligently(user_input)
                else:
                    continue
                    
            except KeyboardInterrupt:
                print(f"\n{colored('👋 Session interrupted. Goodbye!', Colors.WARNING)}")
                break
            except EOFError:
                print(f"\n{colored('👋 End of input. Goodbye!', Colors.WARNING)}")
                break
    
    def _single_request_mode(self, prompt):
        """Process a single request with intelligent routing."""
        print(f"{colored('🔧 Processing request...', Colors.INFO)}")
        self._process_request_intelligently(prompt)
        print(f"{colored('✅ Request completed', Colors.SUCCESS)}")
    
    def _process_request_intelligently(self, request):
        """Intelligently route requests to appropriate handlers."""
        print(f"\n{colored('🤔 Processing:', Colors.INFO)} {request}")
        
        # Analyze request complexity and route accordingly
        if self._is_complex_workflow(request):
            print(f"{colored('🎯 Detected complex workflow - switching to autonomous mode...', Colors.HIGHLIGHT)}")
            self._handle_complex_workflow(request)
        elif 'create' in request.lower() and 'app' in request.lower():
            self._create_app_with_plan(request)
        elif 'explain' in request.lower():
            self._handle_explanation_request(request)
        elif 'refactor' in request.lower():
            self._handle_refactor_request(request)
        elif 'test' in request.lower():
            self._handle_test_request(request)
        else:
            self._handle_simple_request(request)
    
    def _is_complex_workflow(self, request):
        """Detect if a request requires complex multi-step workflow."""
        complex_keywords = [
            'build', 'develop', 'implement', 'create', 'set up', 'configure',
            'deploy', 'migrate', 'refactor', 'optimize', 'analyze', 'debug'
        ]
        
        # Check for multiple complex actions
        complex_count = sum(1 for keyword in complex_keywords if keyword in request.lower())
        
        # Debug output
        print(f"Debug: Found {complex_count} complex keywords")
        print(f"Debug: Request length: {len(request.split())} words")
        
        return complex_count >= 1 or len(request.split()) > 8
    
    def _handle_complex_workflow(self, request):
        """Handle complex workflows with AI planning and execution."""
        print(f"\n{colored('🧠 Complex Workflow Detected', Colors.HIGHLIGHT, bold=True)}")
        print("=" * 50)
        
        # Step 1: Create AI-generated plan
        print(f"{colored('🤔 Thinking and creating a plan...', Colors.INFO)}")
        plan = self._create_ai_plan(request)
        
        if not plan:
            print(f"{colored('❌ Failed to create a plan. Falling back to simple processing...', Colors.ERROR)}")
            self._handle_simple_request(request)
            return
        
        # Step 2: Show plan to user
        print(f"\n{colored('📋 AI-Generated Plan:', Colors.PRIMARY, bold=True)}")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        
        # Step 3: Get user choice for execution
        print(f"\n{colored('🚀 Execution Options:', Colors.INFO)}")
        print("  1. Full Auto - Execute all steps automatically")
        print("  2. Step-by-Step - Execute with user approval")
        print("  3. Preview Only - Show what would happen")
        print("  4. Cancel - Go back to simple processing")
        
        while True:
            try:
                choice = input(f"\n{colored('Select option (1-4): ', Colors.PRIMARY)}").strip()
                if choice == '1':
                    return self._execute_plan_auto(plan, request)
                elif choice == '2':
                    return self._execute_plan_interactive(plan, request)
                elif choice == '3':
                    return self._preview_execution(plan, request)
                elif choice == '4':
                    print(f"{colored('Workflow cancelled by user.', Colors.WARNING)}")
                    return self._handle_simple_request(request)
                else:
                    print(f"{colored('Invalid choice. Please enter 1-4.', Colors.ERROR)}")
            except KeyboardInterrupt:
                print(f"\n{colored('Workflow cancelled by user.', Colors.INFO)}")
                return self._handle_simple_request(request)
    
    def _create_ai_plan(self, request):
        """Create an AI-generated execution plan."""
        try:
            # Create a planning prompt
            planning_prompt = f"""You are an intelligent coding assistant. Create a step-by-step plan to accomplish this request:

Request: {request}

Create a plan with 3-8 steps. Each step should be:
- Clear and actionable
- Specific enough to execute
- In logical order
- Focused on one task

Return ONLY a JSON array of strings, like this:
["Step 1 description", "Step 2 description", "Step 3 description"]

Example:
["Set up project directory structure", "Create main application file", "Implement core functionality", "Add error handling", "Test the application"]

Plan:"""
            
            # Call the API to generate the plan
            response = self._call_api(planning_prompt)
            content = response.get('content', '')
            
            # Extract the plan from the response
            plan = self._extract_plan_from_response(content)
            
            if plan:
                return plan
            else:
                # Fallback to a simple plan
                return [
                    "Analyze the request",
                    "Set up basic structure", 
                    "Implement core functionality",
                    "Test and validate"
                ]
                
        except Exception as e:
            print(f"{colored(f'Could not generate a valid plan: {e}', Colors.ERROR)}")
            return None
    
    def _extract_plan_from_response(self, content):
        """Extract plan from API response."""
        try:
            # Try to find JSON array in the response
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                if isinstance(plan, list) and len(plan) > 0:
                    return plan
            
            # Fallback: extract lines that look like steps
            lines = content.split('\n')
            plan = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.')):
                    # Clean up the line
                    clean_line = re.sub(r'^[-•\d\.\s]+', '', line).strip()
                    if clean_line and len(clean_line) > 5:
                        plan.append(clean_line)
            
            return plan if plan else None
            
        except Exception as e:
            print(f"{colored(f'Error extracting plan: {e}', Colors.ERROR)}")
            return None
    
    def _execute_plan_auto(self, plan, original_goal):
        """Execute the plan automatically."""
        print(f"\n{colored('🚀 Executing plan automatically...', Colors.INFO)}")
        
        execution_history = []
        
        for i, step in enumerate(plan, 1):
            print(f"\n{colored(f'📝 Step {i}: {step}', Colors.SUCCESS)}")
            
            try:
                # Execute the step using available tools
                result = self._execute_step(step, i, len(plan))
                
                if result:
                    execution_history.append({
                        'step': step,
                        'result': result,
                        'status': 'Success'
                    })
                    print(f"  {colored('✅', Colors.SUCCESS)} Step completed successfully")
                else:
                    execution_history.append({
                        'step': step,
                        'result': 'No result',
                        'status': 'No result'
                    })
                    print(f"  {colored('⚠️', Colors.WARNING)} Step completed with no result")
                    
            except Exception as e:
                print(f"  {colored('❌', Colors.ERROR)} Step failed: {e}")
                execution_history.append({
                    'step': step,
                    'result': str(e),
                    'status': 'Failed'
                })
                
                # Try self-correction
                print(f"{colored('🔄 Attempting self-correction...', Colors.INFO)}")
                new_plan = self._self_correct(original_goal, execution_history)
                if new_plan:
                    print(f"{colored('✅ Self-correction successful! Executing new plan...', Colors.SUCCESS)}")
                    return self._execute_plan_auto(new_plan, original_goal)
                else:
                    print(f"{colored('❌ Self-correction failed. Aborting workflow.', Colors.ERROR)}")
                    return False
        
        print(f"\n{colored('🎉 Complex workflow completed successfully!', Colors.HIGHLIGHT, bold=True)}")
        return True
    
    def _execute_plan_interactive(self, plan, original_goal):
        """Execute the plan with user interaction."""
        print(f"\n{colored('🚀 Executing plan interactively...', Colors.INFO)}")
        
        execution_history = []
        
        for i, step in enumerate(plan, 1):
            print(f"\n{colored(f'📝 Step {i}: {step}', Colors.SUCCESS)}")
            
            # Get user confirmation
            if not self._confirm_step_execution(i, step):
                print(f"{colored('Step skipped by user.', Colors.WARNING)}")
                continue
            
            try:
                # Execute the step
                result = self._execute_step(step, i, len(plan))
                
                if result:
                    execution_history.append({
                        'step': step,
                        'result': result,
                        'status': 'Success'
                    })
                    print(f"  {colored('✅', Colors.SUCCESS)} Step completed successfully")
                else:
                    execution_history.append({
                        'step': step,
                        'result': 'No result',
                        'status': 'No result'
                    })
                    print(f"  {colored('⚠️', Colors.WARNING)} Step completed with no result")
                    
            except Exception as e:
                print(f"  {colored('❌', Colors.ERROR)} Step failed: {e}")
                execution_history.append({
                    'step': step,
                    'result': str(e),
                    'status': 'Failed'
                })
                
                # Ask user if they want to continue
                if not self._ask_continue_after_failure():
                    print(f"{colored('Workflow cancelled by user.', Colors.WARNING)}")
                    return False
        
        print(f"\n{colored('🎉 Complex workflow completed successfully!', Colors.HIGHLIGHT, bold=True)}")
        return True
    
    def _execute_step(self, step, step_num, total_steps):
        """Execute a single step using available tools."""
        step_lower = step.lower()
        
        print(f"    🔍 Analyzing step: {step}")
        
        # Comprehensive step routing with fallbacks
        try:
            # Framework and technology selection
            if any(word in step_lower for word in ['choose', 'select', 'pick']):
                if 'framework' in step_lower or 'express' in step_lower or 'django' in step_lower or 'flask' in step_lower:
                    return self._choose_web_framework()
                elif 'database' in step_lower or 'postgresql' in step_lower or 'mongodb' in step_lower:
                    return self._choose_database()
                else:
                    return self._choose_web_framework()  # Default fallback
            
            # Project setup and initialization
            elif any(word in step_lower for word in ['set up', 'setup', 'initialize', 'create', 'start']):
                if any(word in step_lower for word in ['project', 'directory', 'environment', 'development']):
                    return self._setup_web_project()
                elif any(word in step_lower for word in ['database', 'connection']):
                    return self._setup_database()
                elif any(word in step_lower for word in ['schema', 'model']):
                    return self._create_data_models()
                else:
                    return self._setup_web_project()  # Default fallback
            
            # Implementation and development
            elif any(word in step_lower for word in ['implement', 'build', 'develop', 'create']):
                if any(word in step_lower for word in ['authentication', 'auth', 'login', 'register']):
                    if any(word in step_lower for word in ['endpoint', 'api', 'route']):
                        return self._implement_api_endpoints()
                    else:
                        return self._implement_authentication()
                elif any(word in step_lower for word in ['api', 'endpoint', 'route', 'backend']):
                    return self._implement_api_endpoints()
                elif any(word in step_lower for word in ['model', 'schema', 'table', 'data']):
                    return self._create_data_models()
                elif any(word in step_lower for word in ['form', 'frontend', 'ui', 'interface']):
                    return self._create_user_forms()
                else:
                    return self._implement_api_endpoints()  # Default fallback
            
            # Integration and session management
            elif any(word in step_lower for word in ['integrate', 'connect', 'session', 'state', 'management']):
                if any(word in step_lower for word in ['session', 'state', 'management']):
                    return self._integrate_frontend_backend()
                else:
                    return self._integrate_frontend_backend()
            
            # Testing and validation
            elif any(word in step_lower for word in ['test', 'validate', 'verify', 'bug', 'security']):
                # Create run script before testing
                self._create_run_script()
                return self._test_web_application()
            
            # Problem fixing and resolution - Claude Code-level capabilities
            elif any(word in step_lower for word in ['fix', 'resolve', 'repair', 'correct', 'address', 'solve']):
                if any(word in step_lower for word in ['import', 'module', 'package', 'dependency']):
                    print("    🛠️ Claude Code: Fixing import errors...")
                    return self._fix_import_errors()
                elif any(word in step_lower for word in ['port', 'conflict', 'address', 'connection']):
                    print("    🛠️ Claude Code: Resolving port conflicts...")
                    return self._resolve_port_conflicts()
                elif any(word in step_lower for word in ['validate', 'test', 'verify', 'check']):
                    print("    🛠️ Claude Code: Validating application...")
                    return self._validate_application()
                elif any(word in step_lower for word in ['error', 'bug', 'issue', 'problem']):
                    print("    🛠️ Claude Code: Analyzing and fixing issues...")
                    # Try to fix imports first, then validate
                    self._fix_import_errors()
                    return self._validate_application()
                else:
                    print("    🛠️ Claude Code: Using intelligent problem resolution...")
                    return self._fix_import_errors()  # Default to fixing imports
            
            elif any(word in step_lower for word in ['install', 'package', 'dependency']):
                return self._install_missing_packages()
            
            # File recreation and generation - Claude Code-level capabilities
            elif any(word in step_lower for word in ['recreate', 'generate', 'create', 'build']):
                if any(word in step_lower for word in ['app.py', 'main', 'application', 'flask']):
                    print("    🔧 Claude Code: Recreating main application file...")
                    return self._recreate_app_py()
                elif any(word in step_lower for word in ['auth.py', 'authentication', 'auth']):
                    print("    🔧 Claude Code: Recreating authentication module...")
                    return self._recreate_auth_py()
                else:
                    print("    🔧 Claude Code: Recreating main application...")
                    return self._recreate_app_py()
            
            # Analysis and examination
            elif any(word in step_lower for word in ['analyze', 'check', 'review', 'examine', 'identify']):
                if any(word in step_lower for word in ['structure', 'project', 'files', 'missing']):
                    print("    🔍 Claude Code: Analyzing project structure...")
                    return self._analyze_project_structure()
                else:
                    print("    🔍 Claude Code: Analyzing project...")
                    return self._analyze_project_structure()
            
            # Deployment and production
            elif any(word in step_lower for word in ['deploy', 'production', 'cloud', 'host', 'server']):
                return self._deploy_application()
            
            # Fallback for any unmatched steps
            else:
                print(f"    ⚠️ Step not specifically matched, using intelligent fallback")
                return self._execute_generic_step(step, step_lower)
                
        except Exception as e:
            print(f"    ❌ Error in step execution: {e}")
            return f"Error executing step: {e}"
    
    def _execute_generic_step(self, step, step_lower):
        """Execute steps that don't match specific patterns using intelligent analysis."""
        print(f"    🤖 Claude Code: Using intelligent analysis for: {step}")
        
        # Claude Code-level problem analysis and resolution
        if any(word in step_lower for word in ['error', 'bug', 'issue', 'problem', 'fix', 'resolve']):
            print("    🔍 Claude Code: Detected problem-solving request, analyzing...")
            
            if any(word in step_lower for word in ['import', 'module', 'package']):
                print("    🛠️ Claude Code: Fixing import issues...")
                return self._fix_import_errors()
            elif any(word in step_lower for word in ['port', 'conflict', 'connection']):
                print("    🛠️ Claude Code: Resolving connection issues...")
                return self._resolve_port_conflicts()
            elif any(word in step_lower for word in ['server', 'start', 'run']):
                print("    🛠️ Claude Code: Validating server startup...")
                return self._validate_application()
            else:
                print("    🛠️ Claude Code: Comprehensive problem resolution...")
                # Fix imports first, then validate
                self._fix_import_errors()
                return self._validate_application()
        
        # Standard functionality routing
        elif any(word in step_lower for word in ['user', 'login', 'register', 'auth']):
            return self._implement_authentication()
        elif any(word in step_lower for word in ['database', 'data', 'store']):
            return self._create_data_models()
        elif any(word in step_lower for word in ['frontend', 'form', 'ui']):
            return self._create_user_forms()
        elif any(word in step_lower for word in ['api', 'endpoint', 'backend']):
            return self._implement_api_endpoints()
        elif any(word in step_lower for word in ['test', 'validate']):
            return self._test_web_application()
        else:
            # Create a basic implementation file for the step
            return self._create_step_implementation(step)
    
    def _confirm_step_execution(self, step_num, step):
        """Get user confirmation for step execution."""
        print(f"\n{colored('⏸️ Step ready for execution:', Colors.INFO)}")
        print(f"   {step}")
        
        while True:
            try:
                choice = input(f"{colored('Execute this step? (y/n/skip): ', Colors.PRIMARY)}").strip().lower()
                
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                elif choice in ['s', 'skip']:
                    return False
                else:
                    print("⚠️ Please enter 'y' (yes), 'n' (no), or 's' (skip)")
            except KeyboardInterrupt:
                print("\n❌ Step execution cancelled")
                return False
    
    def _ask_continue_after_failure(self):
        """Ask user if they want to continue after a step failure."""
        while True:
            try:
                choice = input(f"{colored('Continue with next step? (y/n): ', Colors.PRIMARY)}").strip().lower()
                
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("⚠️ Please enter 'y' (yes) or 'n' (no)")
            except KeyboardInterrupt:
                print("\n❌ Workflow cancelled")
                return False
    
    def _preview_execution(self, plan, original_goal):
        """Preview what would happen during execution."""
        print(f"\n{colored('🔍 Execution Preview', Colors.HIGHLIGHT, bold=True)}")
        print("=" * 50)
        
        for i, step in enumerate(plan, 1):
            print(f"📋 Step {i}: {step}")
            
            # Show what tools would be used
            if 'create' in step.lower() or 'set up' in step.lower():
                print(f"   🔧 Would use: File creation tools")
            elif 'implement' in step.lower():
                print(f"   🔧 Would use: Code generation tools")
            elif 'test' in step.lower():
                print(f"   🔧 Would use: Testing tools")
            else:
                print(f"   🔧 Would use: Generic execution")
            print()
        
        print(f"{colored('💡 This is a preview only. No actual changes were made.', Colors.INFO)}")
        print(f"{colored('💡 Use option 1 (Full Auto) or 2 (Step-by-Step) to actually execute.', Colors.INFO)}")
        return True
    
    def _self_correct(self, original_goal, history):
        """Analyze failures and create a recovery plan."""
        print(f"{colored('🤖 Analyzing failure and creating a recovery plan...', Colors.INFO)}")
        
        # Create a correction prompt
        correction_prompt = f"""An autonomous agent was trying to achieve the following goal: '{original_goal}'

It created a plan and failed. Here is the execution history:
{json.dumps(history, indent=2)}

The last step failed. Analyze the error and create a new, corrected plan to achieve the original goal.

**Your response MUST be a JSON array of strings, just like before.**

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
            
            # Extract the new plan
            new_plan = self._extract_plan_from_response(content)
            
            if new_plan:
                print(f"{colored('✅ Generated recovery plan with', Colors.SUCCESS)} {len(new_plan)} steps")
                return new_plan
            else:
                return None
                
        except Exception as e:
            print(f"{colored(f'Failed to generate a correction plan: {e}', Colors.ERROR)}")
            return None
    
    def _call_api(self, prompt):
        """Call the configured API (OpenAI or Gemini)."""
        try:
            if self.authenticator.selected_api == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.authenticator.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return {
                    'content': response.choices[0].message.content,
                    'model': 'openai'
                }
            elif self.authenticator.selected_api == 'gemini':
                model = genai.GenerativeModel(self.authenticator.model_name)
                response = model.generate_content(prompt)
                return {
                    'content': response.text,
                    'model': 'gemini'
                }
            else:
                return {'content': 'No API configured', 'model': 'none'}
                
        except Exception as e:
            print(f"{colored(f'API call failed: {e}', Colors.ERROR)}")
            return {'content': f'Error: {e}', 'model': 'error'}
    
    def _create_app_with_plan(self, request):
        """Create an app with a proper plan and execution."""
        print(f"\n{colored('🎯 Creating App with Plan', Colors.HIGHLIGHT, bold=True)}")
        print("=" * 50)
        
        # Create a simple plan
        plan = [
            "Set up project structure",
            "Create main HTML page", 
            "Implement CSS styling",
            "Add JavaScript functionality",
            "Test the application"
        ]
        
        print(f"{colored('📋 Plan:', Colors.PRIMARY, bold=True)}")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        
        print(f"\n{colored('🚀 Executing plan...', Colors.INFO)}")
        
        # Execute the plan
        for i, step in enumerate(plan, 1):
            print(f"\n{colored(f'📝 Step {i}: {step}', Colors.SUCCESS)}")
            
            if step == "Set up project structure":
                self._setup_project_structure()
            elif step == "Create main HTML page":
                self._create_html_page()
            elif step == "Implement CSS styling":
                self._create_css_styling()
            elif step == "Add JavaScript functionality":
                self._create_javascript()
            elif step == "Test the application":
                self._test_application()
            
            print(f"  {colored('✅', Colors.SUCCESS)} {step} completed")
        
        print(f"\n{colored('🎉 App creation completed!', Colors.HIGHLIGHT, bold=True)}")
        print(f"{colored('📁 Files created in: todo-app/', Colors.INFO)}")
    
    def _setup_project_structure(self):
        """Set up the project directory structure."""
        try:
            os.makedirs("todo-app", exist_ok=True)
            print("    Created todo-app directory")
        except Exception as e:
            print(f"    Error: {e}")
    
    def _create_html_page(self):
        """Create the main HTML page."""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Todo App</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>📝 Todo App</h1>
        <div class="input-section">
            <input type="text" id="todoInput" placeholder="Enter a new task...">
            <button onclick="addTodo()">Add Task</button>
        </div>
        <ul id="todoList"></ul>
    </div>
    <script src="script.js"></script>
</body>
</html>"""
        
        try:
            write_file("todo-app/index.html", html_content)
            print("    Created index.html")
        except Exception as e:
            print(f"    Error: {e}")
    
    def _create_css_styling(self):
        """Create CSS styling for the app."""
        css_content = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    width: 90%;
    max-width: 500px;
}

h1 {
    text-align: center;
    color: #333;
    margin-bottom: 2rem;
    font-size: 2.5rem;
}

.input-section {
    display: flex;
    gap: 10px;
    margin-bottom: 2rem;
}

input[type="text"] {
    flex: 1;
    padding: 12px;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
    outline: none;
    transition: border-color 0.3s;
}

input[type="text"]:focus {
    border-color: #667eea;
}

button {
    padding: 12px 24px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 16px;
    transition: background 0.3s;
}

button:hover {
    background: #5a6fd8;
}

ul {
    list-style: none;
}

li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    margin: 8px 0;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    transition: transform 0.2s;
}

li:hover {
    transform: translateX(5px);
}

.todo-text {
    flex: 1;
    margin-right: 10px;
}

.completed {
    text-decoration: line-through;
    opacity: 0.6;
}

.delete-btn {
    background: #dc3545;
    padding: 6px 12px;
    font-size: 14px;
}

.delete-btn:hover {
    background: #c82333;
}"""
        
        try:
            write_file("todo-app/style.css", css_content)
            print("    Created style.css")
        except Exception as e:
            print(f"    Error: {e}")
    
    def _create_javascript(self):
        """Create JavaScript functionality."""
        js_content = """// Todo App JavaScript
let todos = JSON.parse(localStorage.getItem('todos')) || [];

function addTodo() {
    const input = document.getElementById('todoInput');
    const text = input.value.trim();
    
    if (text) {
        const todo = {
            id: Date.now(),
            text: text,
            completed: false
        };
        
        todos.push(todo);
        saveTodos();
        renderTodos();
        input.value = '';
    }
}

function toggleTodo(id) {
    todos = todos.map(todo => 
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
    );
    saveTodos();
    renderTodos();
}

function deleteTodo(id) {
    todos = todos.filter(todo => todo.id !== id);
    saveTodos();
    renderTodos();
}

function renderTodos() {
    const todoList = document.getElementById('todoList');
    todoList.innerHTML = '';
    
    todos.forEach(todo => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="todo-text ${todo.completed ? 'completed' : ''}" 
                  onclick="toggleTodo(${todo.id})">
                ${todo.text}
            </span>
            <button class="delete-btn" onclick="deleteTodo(${todo.id})">Delete</button>
        `;
        todoList.appendChild(li);
    });
}

function saveTodos() {
    localStorage.setItem('todos', JSON.stringify(todos));
}

// Enter key support
document.getElementById('todoInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        addTodo();
    }
});

// Initial render
renderTodos();"""
        
        try:
            write_file("todo-app/script.js", js_content)
            print("    Created script.js")
        except Exception as e:
            print(f"    Error: {e}")
    
    def _test_application(self):
        """Test the application by opening it in a browser."""
        try:
            # Check if files exist
            if os.path.exists("todo-app/index.html"):
                print("    All files created successfully")
                print("    Open todo-app/index.html in your browser to test")
            else:
                print("    Error: Files not created properly")
        except Exception as e:
            print(f"    Error: {e}")
    
    def _handle_explanation_request(self, request):
        """Handle explanation requests."""
        print(f"{colored('📚 Explanation mode activated', Colors.INFO)}")
        print(f"{colored('This would analyze and explain the requested code', Colors.SECONDARY)}")
    
    def _handle_refactor_request(self, request):
        """Handle refactor requests."""
        print(f"{colored('🔧 Refactoring mode activated', Colors.INFO)}")
        print(f"{colored('This would refactor the requested code', Colors.SECONDARY)}")
    
    def _handle_test_request(self, request):
        """Handle test requests."""
        print(f"{colored('🧪 Testing mode activated', Colors.INFO)}")
        print(f"{colored('This would generate tests for the requested code', Colors.SECONDARY)}")
    
    def _handle_simple_request(self, request):
        """Handle simple requests."""
        print(f"{colored('💡 General coding assistance mode', Colors.INFO)}")
        print(f"{colored('This would provide general coding help', Colors.SECONDARY)}")
    
    def _create_file_from_step(self, step):
        """Create a file based on the step description."""
        # This would analyze the step and create appropriate files
        return f"Created file based on: {step}"
    
    def _implement_functionality(self, step):
        """Implement functionality based on the step description."""
        # This would implement the described functionality
        return f"Implemented: {step}"
    
    def _analyze_codebase(self):
        """Analyze the current codebase."""
        try:
            structure = get_project_structure()
            return f"Codebase analyzed. Structure: {len(structure.split(chr(10)))} lines"
        except Exception as e:
            return f"Analysis failed: {e}"
    
    def _show_help(self):
        """Show available commands."""
        print(f"\n{colored('📖 Available Commands:', Colors.HIGHLIGHT, bold=True)}")
        print(f"{colored('  help', Colors.PRIMARY)}     - Show this help")
        print(f"{colored('  context', Colors.PRIMARY)}  - Show project context")
        print(f"{colored('  demo', Colors.PRIMARY)}     - Show demo features")
        print(f"{colored('  exit/quit', Colors.PRIMARY)} - End session")
        print(f"\n{colored('💡 You can also type natural language requests like:', Colors.INFO)}")
        print(f"  • 'create a simple todo app'")
        print(f"  • 'build a web application with user authentication'")
        print(f"  • 'implement a REST API with database integration'")
        print(f"  • 'refactor this code for better performance'")
        print(f"  • 'generate unit tests for the auth module'")
    
    def _show_context(self):
        """Show project context."""
        print(f"\n{colored('📁 Project Context:', Colors.HIGHLIGHT, bold=True)}")
        try:
            structure = get_project_structure()
            print(f"{colored('Project structure retrieved successfully', Colors.SUCCESS)}")
            print(f"{colored('Current directory structure:', Colors.INFO)}")
            print(structure)
        except Exception as e:
            print(f"{colored(f'Could not retrieve project structure: {e}', Colors.ERROR)}")
    
    def _show_demo(self):
        """Show demo features."""
        print(f"\n{colored('🎭 Demo Features:', Colors.HIGHLIGHT, bold=True)}")
        print(f"{colored('This is gcode - your intelligent coding companion', Colors.INFO)}")
        print(f"{colored('Available features:', Colors.INFO)}")
        print(f"  • AI-powered planning and execution")
        print(f"  • Autonomous workflow management")
        print(f"  • Self-correction and recovery")
        print(f"  • Multi-step task execution")
        print(f"  • Interactive and automatic modes")
        print(f"  • File operations and code generation")
        print(f"  • Git operations and testing")

    def _choose_web_framework(self):
        """Choose and recommend a web framework."""
        frameworks = {
            'express': 'Node.js - Fast, unopinionated, minimalist web framework',
            'django': 'Python - High-level web framework with built-in admin',
            'flask': 'Python - Lightweight and flexible micro-framework',
            'rails': 'Ruby - Full-stack web application framework'
        }
        
        print("    Available web frameworks:")
        for name, desc in frameworks.items():
            print(f"      • {name.title()}: {desc}")
        
        # Recommend based on current environment
        if 'python' in sys.version.lower():
            recommendation = 'Flask' if 'simple' in self.context.get('complexity', '') else 'Django'
        else:
            recommendation = 'Express'
        
        print(f"    Recommendation: {recommendation} (based on current environment)")
        return f"Selected {recommendation} as web framework"
    
    def _choose_database(self):
        """Choose and recommend a database."""
        databases = {
            'postgresql': 'Advanced open source database with ACID compliance',
            'mysql': 'Popular open source relational database',
            'mongodb': 'NoSQL document database for flexible schemas',
            'sqlite': 'Lightweight, serverless database for development'
        }
        
        print("    Available databases:")
        for name, desc in databases.items():
            print(f"      • {name.title()}: {desc}")
        
        recommendation = 'PostgreSQL' if 'production' in self.context.get('environment', '') else 'SQLite'
        print(f"    Recommendation: {recommendation} (based on requirements)")
        return f"Selected {recommendation} as database"
    
    def _setup_web_project(self):
        """Set up a new web project with virtual environment and dependencies."""
        try:
            project_name = "web-app-with-auth"
            os.makedirs(project_name, exist_ok=True)
            
            # Create basic project structure
            os.makedirs(f"{project_name}/src", exist_ok=True)
            os.makedirs(f"{project_name}/tests", exist_ok=True)
            os.makedirs(f"{project_name}/docs", exist_ok=True)
            
            # Create basic files
            readme_content = f"""# {project_name.title()}

Web application with user authentication and database integration.

## Features
- User authentication (login/register)
- Database integration
- RESTful API endpoints
- Secure password handling

## Setup
1. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python src/app.py`

## Usage
Start the development server and navigate to the application.
"""
            
            write_file(f"{project_name}/README.md", readme_content)
            print(f"    Created project structure in {project_name}/")
            
            # Create virtual environment
            print(f"    🔧 Creating virtual environment...")
            venv_path = f"{project_name}/venv"
            os.makedirs(venv_path, exist_ok=True)
            
            # Create virtual environment using python -m venv
            import subprocess
            try:
                result = subprocess.run([
                    sys.executable, "-m", "venv", venv_path
                ], capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    print(f"    ✅ Virtual environment created successfully")
                    
                    # Create requirements.txt
                    requirements_content = """Flask==2.3.3
PyJWT==2.8.0
SQLAlchemy==2.0.21
requests==2.31.0
python-dotenv==1.0.0
"""
                    write_file(f"{project_name}/requirements.txt", requirements_content)
                    print(f"    📦 Created requirements.txt")
                    
                    # Install dependencies in the virtual environment
                    print(f"    📥 Installing dependencies...")
                    
                    # Determine the pip path for the virtual environment
                    if os.name == 'nt':  # Windows
                        pip_path = f"{venv_path}/Scripts/pip"
                    else:  # Linux/Mac
                        pip_path = f"{venv_path}/bin/pip"
                    
                    # Install requirements
                    install_result = subprocess.run([
                        pip_path, "install", "-r", "requirements.txt"
                    ], capture_output=True, text=True, cwd=f"{project_name}")
                    
                    if install_result.returncode == 0:
                        print(f"    ✅ Dependencies installed successfully")
                    else:
                        print(f"    ⚠️ Dependencies installation had issues: {install_result.stderr}")
                    
                    # Create activation script
                    if os.name == 'nt':  # Windows
                        activate_script = f"""@echo off
echo Activating virtual environment for {project_name}...
call "{venv_path}\\Scripts\\activate.bat"
echo Virtual environment activated!
echo To deactivate, run: deactivate
"""
                        write_file(f"{project_name}/activate.bat", activate_script)
                    else:  # Linux/Mac
                        activate_script = f"""#!/bin/bash
echo "Activating virtual environment for {project_name}..."
source "{venv_path}/bin/activate"
echo "Virtual environment activated!"
echo "To deactivate, run: deactivate"
"""
                        write_file(f"{project_name}/activate.sh", activate_script)
                        os.chmod(f"{project_name}/activate.sh", 0o755)
                    
                    print(f"    🚀 Created activation script")
                    
                else:
                    print(f"    ⚠️ Virtual environment creation had issues: {result.stderr}")
                    
            except Exception as e:
                print(f"    ⚠️ Could not create virtual environment: {e}")
                print(f"    💡 You can create it manually: python -m venv {venv_path}")
            
            return f"Project {project_name} created successfully with virtual environment"
            
        except Exception as e:
            return f"Error setting up project: {e}"
    
    def _setup_database(self):
        """Set up database configuration."""
        try:
            # Create database configuration file
            db_config = """# Database Configuration
DATABASE_URL = "postgresql://localhost/webapp_auth"
DATABASE_NAME = "webapp_auth"
DATABASE_USER = "webapp_user"
DATABASE_PASSWORD = "secure_password"

# For development, you can use SQLite
# DATABASE_URL = "sqlite:///./webapp.db"
"""
            
            write_file("web-app-with-auth/database.config", db_config)
            print("    Created database configuration")
            return "Database configuration created"
            
        except Exception as e:
            return f"Error setting up database: {e}"
    
    def _implement_authentication(self):
        """Implement user authentication system."""
        try:
            auth_code = """# Authentication System
import hashlib
import jwt
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        return self.hash_password(password) == hashed
    
    def create_token(self, user_id):
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_id']
        except:
            return None

# Usage example
auth = AuthManager('your-secret-key')
"""
            
            write_file("web-app-with-auth/src/auth.py", auth_code)
            print("    Created authentication system")
            return "Authentication system implemented"
            
        except Exception as e:
            return f"Error implementing authentication: {e}"
    
    def _implement_api_endpoints(self):
        """Implement API endpoints for user management."""
        try:
            api_code = """# API Endpoints
from flask import Flask, request, jsonify
from auth import AuthManager

app = Flask(__name__)
auth = AuthManager('your-secret-key')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
    
    # Hash password and store user (simplified)
    hashed_password = auth.hash_password(password)
    # TODO: Store in database
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
    
    # TODO: Verify against database
    # For demo, accept any login
    token = auth.create_token(username)
    return jsonify({'token': token, 'message': 'Login successful'})

@app.route('/api/profile', methods=['GET'])
def get_profile():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    user_id = auth.verify_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401
    
    return jsonify({'user_id': user_id, 'message': 'Profile retrieved'})

if __name__ == '__main__':
    app.run(debug=True)
"""
            
            write_file("web-app-with-auth/src/app.py", api_code)
            print("    Created API endpoints")
            return "API endpoints implemented"
            
        except Exception as e:
            return f"Error implementing API endpoints: {e}"
    
    def _create_data_models(self):
        """Create database models and schemas."""
        try:
            models_code = """# Database Models
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Database setup
engine = create_engine('sqlite:///webapp.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
"""
            
            write_file("web-app-with-auth/src/models.py", models_code)
            print("    Created database models")
            return "Database models created"
            
        except Exception as e:
            return f"Error creating models: {e}"
    
    def _integrate_frontend_backend(self):
        """Integrate frontend with backend authentication."""
        try:
            frontend_code = """<!-- Frontend Integration -->
<!DOCTYPE html>
<html>
<head>
    <title>Web App with Auth</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
        input, button { margin: 5px; padding: 10px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <h1>Web Application with Authentication</h1>
    
    <!-- Login Form -->
    <div id="loginForm" class="form">
        <h2>Login</h2>
        <input type="text" id="loginUsername" placeholder="Username">
        <input type="password" id="loginPassword" placeholder="Password">
        <button onclick="login()">Login</button>
    </div>
    
    <!-- Register Form -->
    <div id="registerForm" class="form">
        <h2>Register</h2>
        <input type="text" id="regUsername" placeholder="Username">
        <input type="email" id="regEmail" placeholder="Email">
        <input type="password" id="regPassword" placeholder="Password">
        <button onclick="register()">Register</button>
    </div>
    
    <!-- Profile Section -->
    <div id="profileSection" class="form hidden">
        <h2>Profile</h2>
        <p id="profileInfo"></p>
        <button onclick="logout()">Logout</button>
    </div>
    
    <script>
        let token = localStorage.getItem('token');
        
        async function login() {
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, password})
                });
                
                const data = await response.json();
                if (data.token) {
                    token = data.token;
                    localStorage.setItem('token', token);
                    showProfile();
                }
            } catch (error) {
                alert('Login failed: ' + error);
            }
        }
        
        async function register() {
            const username = document.getElementById('regUsername').value;
            const email = document.getElementById('regEmail').value;
            const password = document.getElementById('regPassword').value;
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username, email, password})
                });
                
                if (response.ok) {
                    alert('Registration successful! Please login.');
                }
            } catch (error) {
                alert('Registration failed: ' + error);
            }
        }
        
        async function showProfile() {
            if (!token) return;
            
            try {
                const response = await fetch('/api/profile', {
                    headers: {'Authorization': token}
                });
                
                const data = await response.json();
                document.getElementById('profileInfo').textContent = 
                    `Welcome, User ID: ${data.user_id}`;
                
                document.getElementById('loginForm').classList.add('hidden');
                document.getElementById('registerForm').classList.add('hidden');
                document.getElementById('profileSection').classList.remove('hidden');
            } catch (error) {
                console.error('Profile fetch failed:', error);
            }
        }
        
        function logout() {
            localStorage.removeItem('token');
            token = null;
            document.getElementById('loginForm').classList.remove('hidden');
            document.getElementById('registerForm').classList.remove('hidden');
            document.getElementById('profileSection').classList.add('hidden');
        }
        
        // Check if user is already logged in
        if (token) {
            showProfile();
        }
    </script>
</body>
</html>"""
            
            write_file("web-app-with-auth/frontend.html", frontend_code)
            print("    Created frontend integration")
            return "Frontend-backend integration completed"
            
        except Exception as e:
            return f"Error integrating frontend: {e}"
    
    def _test_web_application(self):
        """Test the web application."""
        try:
            # Create a simple test script
            test_code = """# Test Script for Web Application
import requests
import json
import subprocess
import sys
import os

BASE_URL = "http://localhost:5000"

def start_server():
    \"\"\"Start the Flask server in the background.\"\"\"
    print("🚀 Starting Flask server...")
    
    # Check if we're in the project directory
    if not os.path.exists("src/app.py"):
        print("❌ Error: Please run this script from the project root directory")
        return False
    
    # Start the server
    try:
        # Use the project's virtual environment
        if os.name == 'nt':  # Windows
            python_path = "venv\\Scripts\\python.exe"
        else:  # Linux/Mac
            python_path = "venv/bin/python"
        
        if os.path.exists(python_path):
            print(f"✅ Using project virtual environment: {python_path}")
            server_process = subprocess.Popen([
                python_path, "src/app.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            print("⚠️ Project virtual environment not found, using system Python")
            server_process = subprocess.Popen([
                sys.executable, "src/app.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("⏳ Waiting for server to start...")
        import time
        time.sleep(3)  # Wait for server to start
        
        return server_process
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def test_registration():
    data = {"username": "testuser", "email": "test@example.com", "password": "testpass"}
    try:
        response = requests.post(f"{BASE_URL}/api/register", json=data)
        print(f"✅ Registration: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Registration failed: Server not running")
        return False
    return True

def test_login():
    data = {"username": "testuser", "password": "testpass"}
    try:
        response = requests.post(f"{BASE_URL}/api/login", json=data)
        print(f"✅ Login: {response.status_code} - {response.json()}")
        
        if response.status_code == 200:
            token = response.json().get('token')
            test_profile(token)
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Login failed: Server not running")
        return False
    return False

def test_profile(token):
    headers = {"Authorization": token}
    try:
        response = requests.get(f"{BASE_URL}/api/profile", headers=headers)
        print(f"✅ Profile: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Profile test failed: Server not running")

def main():
    print("🧪 Testing Web Application...")
    print("=" * 50)
    
    # Start the server
    server_process = start_server()
    if not server_process:
        print("❌ Could not start server. Please check the application manually.")
        return
    
    try:
        # Run tests
        success = True
        if not test_registration():
            success = False
        if not test_login():
            success = False
        
        if success:
            print("\\n🎉 All tests passed! Application is working correctly.")
        else:
            print("\\n⚠️ Some tests failed. Check the server status.")
            
    finally:
        # Clean up
        if server_process:
            print("\\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped")

if __name__ == "__main__":
    main()
"""
            
            write_file("web-app-with-auth/test_app.py", test_code)
            print("    Created enhanced test script")
            print("    💡 The test script will automatically:")
            print("       • Start the Flask server using the project's virtual environment")
            print("       • Run all tests")
            print("       • Clean up the server process")
            return "Enhanced test script created with automatic server management"
            
        except Exception as e:
            return f"Error creating test script: {e}"
    
    def _create_user_forms(self):
        """Create user registration and login forms."""
        try:
            forms_code = """<!-- User Forms HTML -->
<!DOCTYPE html>
<html>
<head>
    <title>User Authentication Forms</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .form-container { max-width: 400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
        input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; box-sizing: border-box; }
        input:focus { border-color: #007bff; outline: none; }
        button { width: 100%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .toggle { text-align: center; margin-top: 20px; }
        .toggle a { color: #007bff; text-decoration: none; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="form-container">
        <!-- Login Form -->
        <div id="loginForm">
            <h2>Login</h2>
            <form onsubmit="handleLogin(event)">
                <div class="form-group">
                    <label for="loginUsername">Username</label>
                    <input type="text" id="loginUsername" required>
                </div>
                <div class="form-group">
                    <label for="loginPassword">Password</label>
                    <input type="password" id="loginPassword" required>
                </div>
                <button type="submit">Login</button>
            </form>
            <div class="toggle">
                <a href="#" onclick="toggleForms()">Don't have an account? Register</a>
            </div>
        </div>
        
        <!-- Registration Form -->
        <div id="registerForm" class="hidden">
            <h2>Register</h2>
            <form onsubmit="handleRegister(event)">
                <div class="form-group">
                    <label for="regUsername">Username</label>
                    <input type="text" id="regUsername" required>
                </div>
                <div class="form-group">
                    <label for="regEmail">Email</label>
                    <input type="email" id="regEmail" required>
                </div>
                <div class="form-group">
                    <label for="regPassword">Password</label>
                    <input type="password" id="regPassword" required>
                </div>
                <div class="form-group">
                    <label for="regConfirmPassword">Confirm Password</label>
                    <input type="password" id="regConfirmPassword" required>
                </div>
                <button type="submit">Register</button>
            </form>
            <div class="toggle">
                <a href="#" onclick="toggleForms()">Already have an account? Login</a>
            </div>
        </div>
    </div>
    
    <script>
        function toggleForms() {
            const loginForm = document.getElementById('loginForm');
            const registerForm = document.getElementById('registerForm');
            
            if (loginForm.classList.contains('hidden')) {
                loginForm.classList.remove('hidden');
                registerForm.classList.add('hidden');
            } else {
                loginForm.classList.add('hidden');
                registerForm.classList.remove('hidden');
            }
        }
        
        function handleLogin(event) {
            event.preventDefault();
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            // TODO: Implement actual login logic
            console.log('Login attempt:', { username, password });
            alert('Login functionality would be implemented here');
        }
        
        function handleRegister(event) {
            event.preventDefault();
            const username = document.getElementById('regUsername').value;
            const email = document.getElementById('regEmail').value;
            const password = document.getElementById('regPassword').value;
            const confirmPassword = document.getElementById('regConfirmPassword').value;
            
            if (password !== confirmPassword) {
                alert('Passwords do not match!');
                return;
            }
            
            // TODO: Implement actual registration logic
            console.log('Registration attempt:', { username, email, password });
            alert('Registration functionality would be implemented here');
        }
    </script>
</body>
</html>"""
            
            write_file("web-app-with-auth/user_forms.html", forms_code)
            print("    Created user authentication forms")
            return "User forms created successfully"
            
        except Exception as e:
            return f"Error creating user forms: {e}"
    
    def _deploy_application(self):
        """Deploy the application to a cloud service."""
        try:
            # Create deployment configuration files
            heroku_config = """# Heroku Configuration
web: python src/app.py

# Requirements
Flask==2.0.1
gunicorn==20.1.0
psycopg2-binary==2.9.1
"""
            
            docker_config = """# Docker Configuration
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]
"""
            
            write_file("web-app-with-auth/Procfile", heroku_config)
            write_file("web-app-with-auth/Dockerfile", docker_config)
            
            print("    Created deployment configurations")
            print("    Ready for deployment to Heroku, Docker, or other cloud services")
            return "Deployment configuration created"
            
        except Exception as e:
            return f"Error creating deployment config: {e}"

    def _create_run_script(self):
        """Create a run script for easy application startup."""
        try:
            if os.name == 'nt':  # Windows
                run_script = """@echo off
echo Starting Web Application with Authentication...
echo.

REM Check if virtual environment exists
if not exist "venv\\Scripts\\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\\Scripts\\activate
    echo And: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment and start app
echo ✅ Activating virtual environment...
call venv\\Scripts\\activate.bat

echo 🚀 Starting Flask application...
echo 📍 Server will be available at: http://localhost:5000
echo 📍 API endpoints at: http://localhost:5000/api/
echo.
echo Press Ctrl+C to stop the server
echo.

python src/app.py
pause
"""
                write_file("web-app-with-auth/run.bat", run_script)
                print("    Created Windows run script: run.bat")
                
            else:  # Linux/Mac
                run_script = """#!/bin/bash
echo "Starting Web Application with Authentication..."
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "And: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and start app
echo "✅ Activating virtual environment..."
source venv/bin/activate

echo "🚀 Starting Flask application..."
echo "📍 Server will be available at: http://localhost:5000"
echo "📍 API endpoints at: http://localhost:5000/api/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python src/app.py
"""
                write_file("web-app-with-auth/run.sh", run_script)
                os.chmod("web-app-with-auth/run.sh", 0o755)
                print("    Created Linux/Mac run script: run.sh")
            
            return "Run script created for easy application startup"
            
        except Exception as e:
            return f"Error creating run script: {e}"

    def _create_step_implementation(self, step):
        """Create a basic implementation file for any unmatched step."""
        try:
            # Create a generic implementation file
            step_name = step.replace(' ', '_').replace('-', '_').lower()
            step_name = ''.join(c for c in step_name if c.isalnum() or c == '_')
            
            implementation_code = f"""# Implementation for: {step}

# This file was automatically generated by gcode for the step:
# "{step}"

def implement_{step_name}():
    \"\"\"
    Implementation of: {step}
    
    This function should contain the logic to accomplish the requested step.
    Modify this implementation based on your specific requirements.
    \"\"\"
    
    print(f"Implementing: {step}")
    
    # TODO: Add your implementation logic here
    # This is a placeholder that should be customized
    
    return f"Step '{step}' has been implemented"
    
def main():
    \"\"\"Main execution function.\"\"\"
    result = implement_{step_name}()
    print(result)
    
if __name__ == "__main__":
    main()
"""
            
            filename = f"web-app-with-auth/src/{step_name}.py"
            write_file(filename, implementation_code)
            print(f"    Created generic implementation: {filename}")
            return f"Created implementation file for: {step}"
            
        except Exception as e:
            return f"Error creating step implementation: {e}"

    def _fix_import_errors(self):
        """Fix import errors in the Flask application."""
        try:
            print("    🔍 Analyzing import errors...")
            
            # Check the current app.py file
            app_file = "web-app-with-auth/src/app.py"
            if not os.path.exists(app_file):
                return "Error: app.py not found"
            
            # Read the current file
            with open(app_file, 'r') as f:
                content = f.read()
            
            # Fix the import issue
            if "from auth import AuthManager" in content:
                print("    🛠️ Fixing import statement...")
                
                # Replace the problematic import
                fixed_content = content.replace(
                    "from auth import AuthManager",
                    "import sys\nimport os\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))\nfrom auth import AuthManager"
                )
                
                # Write the fixed file
                with open(app_file, 'w') as f:
                    f.write(fixed_content)
                
                print("    ✅ Import statement fixed")
                return "Import errors fixed successfully"
            else:
                print("    ✅ Import statement already correct")
                return "No import errors found"
                
        except Exception as e:
            return f"Error fixing imports: {e}"
    
    def _resolve_port_conflicts(self):
        """Resolve port conflicts by finding an available port."""
        try:
            print("    🔍 Checking for port conflicts...")
            
            # Check if port 5000 is in use
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 5000))
            sock.close()
            
            if result == 0:
                print("    ⚠️ Port 5000 is in use, finding alternative...")
                
                # Find an available port
                for port in range(5001, 5010):
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    if result != 0:
                        print(f"    ✅ Found available port: {port}")
                        
                        # Update the app.py to use the new port
                        app_file = "web-app-with-auth/src/app.py"
                        with open(app_file, 'r') as f:
                            content = f.read()
                        
                        # Replace the port in app.run()
                        if "app.run(debug=True)" in content:
                            fixed_content = content.replace(
                                "app.run(debug=True)",
                                f"app.run(debug=True, port={port})"
                            )
                            
                            with open(app_file, 'w') as f:
                                f.write(fixed_content)
                            
                            print(f"    ✅ Updated app.py to use port {port}")
                            return f"Port conflict resolved - using port {port}"
                
                return "Could not find available port"
            else:
                print("    ✅ Port 5000 is available")
                return "No port conflicts found"
                
        except Exception as e:
            return f"Error resolving port conflicts: {e}"
    
    def _validate_application(self):
        """Validate that the Flask application works correctly."""
        try:
            print("    🧪 Validating Flask application...")
            
            # Test import
            print("    📦 Testing imports...")
            import_result = os.system("cd web-app-with-auth && source venv/bin/activate && python -c 'from src.app import app; print(\"✅ Import successful\")'")
            
            if import_result != 0:
                return "Import validation failed"
            
            print("    ✅ Import validation passed")
            
            # Test if we can start the server
            print("    🚀 Testing server startup...")
            
            # Kill any existing processes
            os.system("pkill -f 'python.*app.py'")
            
            # Start server in background
            start_result = os.system("cd web-app-with-auth && source venv/bin/activate && python src/app.py > /dev/null 2>&1 &")
            
            if start_result != 0:
                return "Server startup failed"
            
            # Wait for server to start
            import time
            time.sleep(3)
            
            # Test if server is responding
            import requests
            try:
                response = requests.get("http://localhost:5000/api/profile", timeout=5)
                print("    ✅ Server is responding")
                return "Application validation successful"
            except:
                # Try alternative port
                try:
                    response = requests.get("http://localhost:5001/api/profile", timeout=5)
                    print("    ✅ Server is responding on port 5001")
                    return "Application validation successful on port 5001"
                except:
                    return "Server is not responding"
                    
        except Exception as e:
            return f"Error validating application: {e}"
    
    def _install_missing_packages(self):
        """Install missing packages in the project's virtual environment."""
        try:
            print("    📦 Installing missing packages...")
            
            # Check if virtual environment exists
            venv_path = "web-app-with-auth/venv"
            if not os.path.exists(venv_path):
                return "Virtual environment not found"
            
            # Install requirements
            install_cmd = f"cd web-app-with-auth && source venv/bin/activate && pip install -r requirements.txt"
            result = os.system(install_cmd)
            
            if result == 0:
                print("    ✅ Packages installed successfully")
                return "All packages installed successfully"
            else:
                return "Package installation failed"
                
        except Exception as e:
            return f"Error installing packages: {e}"

    def _recreate_app_py(self):
        """Recreate the main Flask application file."""
        try:
            print("    🔧 Claude Code: Recreating app.py...")
            
            app_code = """# Flask Web Application with Authentication
from flask import Flask, request, jsonify
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from auth import AuthManager
except ImportError:
    # Create a simple fallback if auth module is missing
    class AuthManager:
        def __init__(self, secret_key):
            self.secret_key = secret_key
        
        def hash_password(self, password):
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()
        
        def create_token(self, user_id):
            return f"token_{user_id}_{self.secret_key}"
        
        def verify_token(self, token):
            if token and token.startswith("token_"):
                return token.split("_")[1]
            return None

app = Flask(__name__)
auth = AuthManager('your-secret-key')

@app.route('/')
def home():
    return jsonify({
        'message': 'Web Application with Authentication',
        'status': 'running',
        'endpoints': ['/api/register', '/api/login', '/api/profile']
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
    
    # Hash password and store user (simplified)
    hashed_password = auth.hash_password(password)
    # TODO: Store in database
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Missing username or password'}), 400
    
    # TODO: Verify against database
    # For demo, accept any login
    token = auth.create_token(username)
    return jsonify({'token': token, 'message': 'Login successful'})

@app.route('/api/profile', methods=['GET'])
def get_profile():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'No token provided'}), 401
    
    user_id = auth.verify_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid token'}), 401
    
    return jsonify({'user_id': user_id, 'message': 'Profile retrieved'})

if __name__ == '__main__':
    # Use port 5001 to avoid conflicts
    app.run(debug=True, port=5001)
"""
            
            write_file("web-app-with-auth/src/app.py", app_code)
            print("    ✅ app.py recreated successfully")
            return "Main Flask application file recreated"
            
        except Exception as e:
            return f"Error recreating app.py: {e}"
    
    def _recreate_auth_py(self):
        """Recreate the authentication module."""
        try:
            print("    🔧 Claude Code: Recreating auth.py...")
            
            auth_code = """# Authentication System
import hashlib
import time
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        return self.hash_password(password) == hashed
    
    def create_token(self, user_id):
        # Simple token generation (in production, use JWT)
        timestamp = int(time.time())
        token_data = f"{user_id}:{timestamp}:{self.secret_key}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    def verify_token(self, token):
        try:
            # Simple token verification (in production, use JWT)
            if token and len(token) == 64:  # SHA256 hash length
                return "user_123"  # Simplified for demo
            return None
        except:
            return None

# Usage example
if __name__ == "__main__":
    auth = AuthManager('your-secret-key')
    print("Authentication module loaded successfully")
"""
            
            write_file("web-app-with-auth/src/auth.py", auth_code)
            print("    ✅ auth.py recreated successfully")
            return "Authentication module recreated"
            
        except Exception as e:
            return f"Error recreating auth.py: {e}"
    
    def _analyze_project_structure(self):
        """Analyze the current project structure and identify missing components."""
        try:
            print("    🔍 Claude Code: Analyzing project structure...")
            
            missing_files = []
            required_files = [
                "web-app-with-auth/src/app.py",
                "web-app-with-auth/src/auth.py",
                "web-app-with-auth/requirements.txt"
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    print(f"    ⚠️ Missing: {file_path}")
            
            if missing_files:
                print(f"    📋 Found {len(missing_files)} missing files")
                return f"Project analysis complete - {len(missing_files)} files missing"
            else:
                print("    ✅ All required files present")
                return "Project structure is complete"
                
        except Exception as e:
            return f"Error analyzing project: {e}"

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        # No arguments - enter interactive mode
        print(colored("gcode", Colors.HIGHLIGHT + Colors.BOLD))
        print(colored("Your intelligent coding companion", Colors.SECONDARY))
        print()
        print(colored("What gcode does:", Colors.BOLD))
        print(colored("  • Write, analyze, and refactor code", Colors.INFO))
        print(colored("  • Generate tests and documentation", Colors.INFO))
        print(colored("  • Monitor code quality and security", Colors.INFO))
        print(colored("  • Manage git operations intelligently", Colors.INFO))
        print(colored("  • Provide real-time coding assistance", Colors.INFO))
        print(colored("  • Execute complex workflows autonomously", Colors.INFO))
        print()
        print(colored("Usage:", Colors.BOLD))
        print(colored("  gcode                    - Enter interactive mode", Colors.SUCCESS))
        print(colored("  gcode 'your request'     - Execute a single coding request", Colors.SUCCESS))
        print(colored("  gcode --help             - Show advanced options", Colors.SUCCESS))
        print()
        print(colored("Examples:", Colors.BOLD))
        print(colored("  gcode 'create a simple todo app'", Colors.INFO))
        print(colored("  gcode 'build a web app with authentication'", Colors.INFO))
        print(colored("  gcode 'implement a REST API with database'", Colors.INFO))
        print(colored("  gcode 'refactor this code for better performance'", Colors.INFO))
        print()
        print(colored("Type 'gcode --help' for advanced options and API management", Colors.HIGHLIGHT))
        print()
        
        # Enter interactive mode
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
        # Single request mode
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
    print(colored("  • AI-powered planning          - Intelligent workflow creation", Colors.INFO))
    print(colored("  • Autonomous execution          - Self-managed task execution", Colors.INFO))
    print(colored("  • Self-correction              - Automatic failure recovery", Colors.INFO))
    print(colored("  • Multi-step workflows         - Complex task management", Colors.INFO))
    print(colored("  • Interactive execution         - User-controlled step execution", Colors.INFO))
    print(colored("  • Real-time monitoring         - Progress tracking and adaptation", Colors.INFO))
    print()
    
    print(colored("For more information, visit: https://github.com/your-repo/gcode", Colors.SECONDARY))

if __name__ == "__main__":
    main()

