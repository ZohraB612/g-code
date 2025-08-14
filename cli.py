import argparse
from .agent import GeminiAgent

def main():
    parser = argparse.ArgumentParser(description="A Gemini-powered coding agent.")
    parser.add_argument("prompt", nargs='*', help="The prompt for the agent.")
    
    args = parser.parse_args()
    
    if not args.prompt:
        print("Please provide a prompt. Example: gcode 'create a file named hello.py'")
        return

    full_prompt = " ".join(args.prompt)
    
    agent = GeminiAgent()
    
    # Send the prompt to the agent and start the conversation
    agent.converse(full_prompt)

if __name__ == "__main__":
    main()
