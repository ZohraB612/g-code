import google.generativeai as genai
import os
from .tools import read_file, write_file, run_shell_command, get_project_structure
from dotenv import load_dotenv

load_dotenv()

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
        # Start the chat with an initial system prompt.
        # This prompt instructs the model to return the plan and tools in a single response.
        self.chat = self.model.start_chat(history=[
            {"role": "user", "parts": [
                "You are an expert software engineer and a pair-programmer. "
                "Your goal is to help me with my coding projects. "
                "You have access to a set of tools to interact with the file system and terminal. "
                "In a single response, you must first provide a step-by-step plan in plain text. "
                "After the plan, provide all the necessary tool calls to execute that plan. "
                "I will review the plan and then approve the tool calls one by one."
            ]}
        ])

    def converse(self, prompt: str):
        """Handles the conversation flow using a single API call to get both plan and tools."""
        print("AGENT: Thinking...")
        # 1. Send the prompt and get a single response containing the plan and tool calls.
        response = self.chat.send_message(prompt)

        # 2. Extract the plan (text) and tool calls from the response.
        response_parts = response.candidates[0].content.parts
        plan_text = "".join(part.text for part in response_parts if part.text).strip()
        tool_calls = [part.function_call for part in response_parts if part.function_call]

        # 3. Display the plan for approval.
        if plan_text:
            print("\nAGENT'S PLAN:")
            print(plan_text)
        else:
            print("AGENT: I could not come up with a plan. Please try a different prompt.")
            # If there's no plan, we can't proceed.
            if not tool_calls:
                return

        # 4. If there are tool calls, ask for approval and execute them.
        if tool_calls:
            # General approval for the plan before executing tools.
            approval = input("\nDo you approve this plan and want to proceed with tool execution? (yes/no): ").lower()
            if approval.strip() != 'yes':
                print("Plan denied. Conversation ended.")
                return
            
            # Iterate through and execute each tool call with individual approval.
            for tool_call in tool_calls:
                func_name = tool_call.name
                func_to_call = AVAILABLE_TOOLS.get(func_name)
                
                if not func_to_call:
                    print(f"Error: Unknown tool '{func_name}'")
                    continue

                func_args = dict(tool_call.args)
                print(f"\nAGENT PROPOSES: {func_name}({func_args})")
                
                tool_approval = input("Approve this tool call? (yes/no): ").lower()
                if tool_approval.strip() != 'yes':
                    print("Tool call denied. Aborting task.")
                    return

                print(f"AGENT: Executing {func_name}...")
                result = func_to_call(**func_args)
                print(f"RESULT:\n{result}")

                # Tool execution complete - no need for additional API call
                print(f"\nAGENT: Tool {func_name} completed successfully.")
                
        else:
            print("\nAGENT: Task complete.")

