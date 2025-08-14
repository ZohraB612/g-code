import google.generativeai as genai
import os
import sys
from .tools import read_file, write_file, run_shell_command, get_project_structure
from dotenv import load_dotenv

load_dotenv()

# Color codes for better formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def colored(text, color):
    """Apply color to text if colors are supported."""
    if sys.platform != 'win32' and os.isatty(sys.stdout.fileno()):
        return f"{color}{text}{Colors.END}"
    return text

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the available tools for the agent
tools = [
    read_file,
    write_file,
    run_shell_command,
    get_project_structure
]

# A mapping of tool names to their functions for easy lookup
AVAILABLE_TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell_command": run_shell_command,
    "get_project_structure": get_project_structure
}

class GeminiAgent:
    """A conversational agent powered by the Gemini model."""
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        """Initializes the agent with a model and chat history."""
        self.model = genai.GenerativeModel(model_name=model_name, tools=tools)
        # Start the chat with an improved system prompt for better autonomy
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
                "Remember: You're a coding partner, not just a tool executor. Complete the entire task in one go."
            ]}
        ])
        self.conversation_history = []

    def converse(self, prompt: str, interactive=False):
        """Handles the conversation flow with improved autonomy."""
        if interactive:
            print(colored("\n🤖 Gemini Agent - Interactive Mode", Colors.HEADER + Colors.BOLD))
            print(colored("Type 'exit' or 'quit' to end the session", Colors.CYAN))
            print(colored("Type 'help' for available commands", Colors.CYAN))
            print(colored("-" * 50, Colors.BLUE))
            
            while True:
                try:
                    user_input = input(colored("\n💬 You: ", Colors.GREEN)).strip()
                    
                    if user_input.lower() in ['exit', 'quit']:
                        print(colored("👋 Goodbye! Session ended.", Colors.YELLOW))
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif not user_input:
                        continue
                    
                    self._process_request(user_input)
                    
                except KeyboardInterrupt:
                    print(colored("\n\n👋 Session interrupted. Goodbye!", Colors.YELLOW))
                    break
                except EOFError:
                    print(colored("\n\n👋 End of input. Goodbye!", Colors.YELLOW))
                    break
        else:
            # Single request mode
            self._process_request(prompt)

    def _process_request(self, prompt: str):
        """Process a single user request."""
        print(colored(f"\n🤖 Agent: Processing your request...", Colors.BLUE))
        
        try:
            # Send the prompt and get response
            response = self.chat.send_message(prompt)
            
            # Extract plan and tool calls
            response_parts = response.candidates[0].content.parts
            plan_text = "".join(part.text for part in response_parts if part.text).strip()
            tool_calls = [part.function_call for part in response_parts if part.function_call]
            
            # Display the plan
            if plan_text:
                print(colored(f"\n📋 Plan:", Colors.CYAN + Colors.BOLD))
                print(plan_text)
            
            # Execute tools if needed
            if tool_calls:
                print(colored(f"\n🔧 Executing {len(tool_calls)} tool(s)...", Colors.YELLOW))
                for i, tool_call in enumerate(tool_calls, 1):
                    self._execute_tool(tool_call, i, len(tool_calls))
                
                # Check if we need to continue with more tools
                self._check_for_more_work(prompt, plan_text)
                
                print(colored(f"\n✅ Task completed successfully!", Colors.GREEN + Colors.BOLD))
            else:
                print(colored(f"\n💡 {plan_text if plan_text else 'Request processed.'}", Colors.CYAN))
                
        except Exception as e:
            print(colored(f"\n❌ Error: {str(e)}", Colors.RED))
            print(colored("Please try again or rephrase your request.", Colors.YELLOW))

    def _check_for_more_work(self, original_prompt, executed_plan):
        """Check if we need to continue with more tools to complete the task."""
        # If the plan mentioned multiple steps but we only executed one tool,
        # ask the model to continue
        if "step" in executed_plan.lower() and "step 1" in executed_plan.lower():
            print(colored(f"\n🔄 Checking if more work is needed...", Colors.YELLOW))
            
            try:
                follow_up_prompt = f"Continue with the remaining steps from the plan: {executed_plan}"
                response = self.chat.send_message(follow_up_prompt)
                
                response_parts = response.candidates[0].content.parts
                additional_tool_calls = [part.function_call for part in response_parts if part.function_call]
                
                if additional_tool_calls:
                    print(colored(f"\n🔧 Executing {len(additional_tool_calls)} additional tool(s)...", Colors.YELLOW))
                    for i, tool_call in enumerate(additional_tool_calls, 1):
                        self._execute_tool(tool_call, i, len(additional_tool_calls))
                    
                    print(colored(f"\n✅ Additional work completed!", Colors.GREEN))
                
            except Exception as e:
                print(colored(f"\n⚠️  Could not check for additional work: {str(e)}", Colors.YELLOW))

    def _execute_tool(self, tool_call, current, total):
        """Execute a single tool call with better feedback."""
        func_name = tool_call.name
        func_to_call = AVAILABLE_TOOLS.get(func_name)
        
        if not func_to_call:
            print(colored(f"❌ Unknown tool '{func_name}'", Colors.RED))
            return
        
        func_args = dict(tool_call.args)
        
        # Show progress and tool info
        print(colored(f"\n🔧 [{current}/{total}] {func_name}", Colors.BLUE))
        if func_args:
            print(colored(f"   Args: {func_args}", Colors.CYAN))
        
        try:
            result = func_to_call(**func_args)
            print(colored(f"   ✅ Result: {result}", Colors.GREEN))
        except Exception as e:
            print(colored(f"   ❌ Error: {str(e)}", Colors.RED))

    def _show_help(self):
        """Show available commands and help."""
        help_text = colored("""
Available Commands:
- Type your request normally (e.g., "create a new file called main.py")
- 'help' - Show this help message
- 'exit' or 'quit' - End the session

Examples:
- "Create a Python file with a hello world function"
- "Show me the project structure"
- "Read the contents of agent.py"
- "Run 'ls -la' to see files"
        """, Colors.CYAN)
        print(help_text)

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print(colored("🤖 Gemini Agent CLI", Colors.HEADER + Colors.BOLD))
        print(colored("A Cursor/Claude Code-like coding assistant", Colors.CYAN))
        print(colored("\nUsage:", Colors.BOLD))
        print(colored("  python -m gcode.cli 'your request here'", Colors.GREEN))
        print(colored("  python -m gcode.cli --interactive", Colors.GREEN))
        print(colored("\nExamples:", Colors.BOLD))
        print(colored("  python -m gcode.cli 'create a hello world file'", Colors.CYAN))
        print(colored("  python -m gcode.cli --interactive", Colors.CYAN))
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

