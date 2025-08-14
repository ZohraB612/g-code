import subprocess
import os

# Define the functions the agent can use.
# The docstrings are critical for the LLM to understand the tools.

def read_file(path: str) -> str:
    """
    Reads the content of a file.
    Args:
        path: The path to the file to be read.
    Returns:
        The content of the file.
    """
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at {path}"

def write_file(path: str, content: str):
    """
    Writes content to a file, creating it if it doesn't exist.
    Args:
        path: The path to the file.
        content: The content to be written to the file.
    Returns:
        A confirmation message.
    """
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to file at {path}"
    except Exception as e:
        return f"Error writing to file: {e}"

def run_shell_command(command: str) -> str:
    """
    Executes a shell command and returns its output.
    Args:
        command: The shell command to be executed.
    Returns:
        The stdout and stderr of the command.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        return output
    except Exception as e:
        return f"Error running command: {e}"

def get_project_structure() -> str:
    """
    Returns a string representing the project's file structure.
    This helps the agent understand the project layout.
    """
    try:
        # We can use the 'tree' command or a simple os.walk
        result = subprocess.run(['tree', '-L', '2'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        # Fallback if 'tree' command isn't available
        project_structure = ""
        for dirpath, dirnames, filenames in os.walk('.'):
            # Ignore hidden files and the venv folder
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != 'venv']
            filenames = [f for f in filenames if not f.startswith('.')]
            level = dirpath.replace('.', '').count(os.sep)
            indent = ' ' * 4 * level
            project_structure += f"{indent}{os.path.basename(dirpath)}/\n"
            subindent = ' ' * 4 * (level + 1)
            for f in filenames:
                project_structure += f"{subindent}{f}\n"
        return project_structure
