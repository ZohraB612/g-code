import google.generativeai as genai
import os
import re
from .tools import read_file, write_file, run_shell_command, get_project_structure
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

tools = [
    read_file,
    write_file,
    run_shell_command,
    get_project_structure
]

class GeminiAgent:
    def __init__(self, model_name="gemini-1.5-flash-latest"):
        self.model = genai.GenerativeModel(model_name=model_name, tools=tools)
        self.chat = self.model.start_chat(history=[
            {"role": "user", "parts": [
                "You are an expert software engineer and a pair-programmer. "
                "Your goal is to help me with my coding projects. "
                "You have access to a set of tools to interact with the file system and terminal. "
                "Always make a plan before performing any actions. "
                "The plan must be a clear, numbered list of steps. "
                "Do not execute any tool calls until I explicitly approve the plan."
            ]}
        ])

    def converse(self, prompt: str):
        print("AGENT: Thinking...")
        response = self.chat.send_message(prompt)

        # Check for and display a plan
        if response.text:
            print("\nAGENT'S PLAN:")
            print(response.text)
            
            # Ask the user for approval
            approval = input("\nDo you approve this plan? (yes/no): ").lower()
            if approval != 'yes':
                print("Plan denied. Conversation ended.")
                return

        # Start the execution loop for each step in the plan
        for step in response.text.split('\n'):
            if not step.strip():
                continue
            
            print(f"\nAGENT: Executing step: {step.strip()}")
            
            # Prompt the model to take the next action for this step
            next_action_response = self.chat.send_message(f"Execute step: {step.strip()}")
            
            if next_action_response.tool_calls:
                for tool_call in next_action_response.tool_calls:
                    func_name = tool_call.function.name
                    func_args = dict(tool_call.function.args)

                    print(f"AGENT PROPOSES: {func_name}({func_args})")
                    
                    # Ask for approval for each individual tool call
                    tool_approval = input("Approve this tool call? (yes/no): ").lower()
                    if tool_approval != 'yes':
                        print("Tool call denied. Aborting task.")
                        return

                    # Execute the approved tool call
                    if func_name in globals():
                        result = globals()[func_name](**func_args)
                        print(f"RESULT:\n{result}")

                        # Send the result back to the model
                        final_response = self.chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(function_response=genai.protos.FunctionResponse(name=func_name, response={'content': result}))]
                            )
                        )
                        print("\nAGENT RESPONSE:")
                        print(final_response.text)