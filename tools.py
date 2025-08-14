import subprocess
import os
import ast
import re
import json
from pathlib import Path

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
                project_structure += f"{indent}{f}\n"
        return project_structure

def analyze_python_file(file_path: str) -> str:
    """
    Analyzes a Python file for code quality, structure, and potential improvements.
    Args:
        file_path: The path to the Python file to analyze.
    Returns:
        A detailed analysis report.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        analysis = []
        analysis.append(f"📊 Analysis of {file_path}")
        analysis.append("=" * 50)
        
        # Basic stats
        lines = content.split('\n')
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        empty_lines = len([l for l in lines if not l.strip()])
        
        analysis.append(f"📈 Statistics:")
        analysis.append(f"   Total lines: {total_lines}")
        analysis.append(f"   Code lines: {code_lines}")
        analysis.append(f"   Comment lines: {comment_lines}")
        analysis.append(f"   Empty lines: {empty_lines}")
        analysis.append(f"   Comment ratio: {(comment_lines/code_lines*100):.1f}%" if code_lines > 0 else "N/A")
        
        # Try to parse AST for deeper analysis
        try:
            tree = ast.parse(content)
            
            # Count functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
            
            analysis.append(f"\n🔧 Code Structure:")
            analysis.append(f"   Functions: {len(functions)}")
            analysis.append(f"   Classes: {len(classes)}")
            analysis.append(f"   Imports: {len(imports)}")
            
            # Function details
            if functions:
                analysis.append(f"\n📝 Functions:")
                for func in functions[:5]:  # Show first 5
                    args = [arg.arg for arg in func.args.args]
                    analysis.append(f"   - {func.name}({', '.join(args)})")
                if len(functions) > 5:
                    analysis.append(f"   ... and {len(functions) - 5} more")
            
            # Class details
            if classes:
                analysis.append(f"\n🏗️  Classes:")
                for cls in classes[:3]:  # Show first 3
                    methods = [node for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]
                    analysis.append(f"   - {cls.name} ({len(methods)} methods)")
                if len(classes) > 3:
                    analysis.append(f"   ... and {len(classes) - 3} more")
                    
        except SyntaxError as e:
            analysis.append(f"\n⚠️  Syntax Error: {e}")
        
        # Code quality checks
        analysis.append(f"\n🔍 Code Quality:")
        
        # Check for long lines
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 79]
        if long_lines:
            analysis.append(f"   ⚠️  Long lines (>79 chars): {len(long_lines)} lines")
        
        # Check for TODO/FIXME comments
        todo_pattern = r'#\s*(TODO|FIXME|HACK|XXX)'
        todos = re.findall(todo_pattern, content, re.IGNORECASE)
        if todos:
            analysis.append(f"   📝 TODO items: {len(todos)} found")
        
        # Check for print statements (potential debugging code)
        print_statements = content.count('print(')
        if print_statements > 0:
            analysis.append(f"   🐛 Print statements: {print_statements} found")
        
        # Suggestions
        analysis.append(f"\n💡 Suggestions:")
        if comment_lines < code_lines * 0.1:
            analysis.append("   - Consider adding more documentation")
        if long_lines:
            analysis.append("   - Consider breaking long lines for readability")
        if print_statements > 0:
            analysis.append("   - Consider removing or replacing print statements with proper logging")
        if not functions and not classes:
            analysis.append("   - File appears to be a script - consider organizing into functions")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error analyzing file {file_path}: {e}"

def create_test_file(source_file: str) -> str:
    """
    Creates a basic test file for a given Python source file.
    Args:
        source_file: The path to the source Python file to create tests for.
    Returns:
        A confirmation message with the test file path.
    """
    try:
        # Read the source file to understand what to test
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Parse to find functions and classes
        try:
            tree = ast.parse(content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        except SyntaxError:
            return f"Error: Could not parse {source_file} - syntax error"
        
        # Generate test file path
        source_path = Path(source_file)
        test_dir = source_path.parent / "tests"
        test_file = test_dir / f"test_{source_path.stem}.py"
        
        # Create test directory if it doesn't exist
        test_dir.mkdir(exist_ok=True)
        
        # Generate test content
        test_content = f'''import pytest
import sys
from pathlib import Path

# Add the parent directory to the path to import the source module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the functions/classes to test
try:
    from {source_path.stem} import *
except ImportError as e:
    print(f"Warning: Could not import from {{source_path.stem}}: {{e}}")

'''
        
        # Add test functions for each function found
        for func in functions:
            test_content += f'''
def test_{func.name}():
    """Test the {func.name} function."""
    # TODO: Add proper test cases
    # Example:
    # result = {func.name}()
    # assert result is not None
    pass
'''
        
        # Add test classes for each class found
        for cls in classes:
            test_content += f'''
class Test{cls.name}:
    """Test the {cls.name} class."""
    
    def test_{cls.name}_creation(self):
        """Test that {cls.name} can be instantiated."""
        # TODO: Add proper test cases
        # Example:
        # instance = {cls.name}()
        # assert instance is not None
        pass
'''
        
        # Write the test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        return f"Successfully created test file at {test_file}\nGenerated {len(functions)} function tests and {len(classes)} class tests."
        
    except Exception as e:
        return f"Error creating test file: {e}"

def search_code(query: str, directory: str = ".") -> str:
    """
    Searches for code patterns, functions, or text across Python files in a directory.
    Args:
        query: The search query (function name, class name, or text to search for).
        directory: The directory to search in (default: current directory).
    Returns:
        Search results with file locations and context.
    """
    try:
        results = []
        query_lower = query.lower()
        
        # Search through Python files
        for py_file in Path(directory).rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Search for matches
                matches = []
                for i, line in enumerate(lines, 1):
                    if query_lower in line.lower():
                        # Get context (previous and next line)
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 2)
                        context_lines = lines[context_start:context_end]
                        
                        matches.append({
                            'line': i,
                            'context': context_lines,
                            'highlight': line.strip()
                        })
                
                if matches:
                    results.append({
                        'file': str(py_file),
                        'matches': matches
                    })
                    
            except Exception as e:
                continue  # Skip files we can't read
        
        if not results:
            return f"No matches found for '{query}' in Python files."
        
        # Format results
        output = [f"🔍 Search results for '{query}':"]
        output.append("=" * 50)
        
        for result in results[:10]:  # Limit to first 10 files
            output.append(f"\n📁 {result['file']}:")
            for match in result['matches'][:5]:  # Limit to first 5 matches per file
                output.append(f"   Line {match['line']}: {match['highlight']}")
                # Show context
                for ctx_line in match['context']:
                    if ctx_line.strip():
                        output.append(f"      {ctx_line.strip()}")
        
        if len(results) > 10:
            output.append(f"\n... and {len(results) - 10} more files with matches")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching code: {e}"

def install_dependencies(requirements_file: str = "requirements.txt") -> str:
    """
    Installs Python dependencies from a requirements file.
    Args:
        requirements_file: Path to the requirements file (default: requirements.txt).
    Returns:
        Installation status and output.
    """
    try:
        if not Path(requirements_file).exists():
            return f"Error: Requirements file {requirements_file} not found."
        
        # Install dependencies
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True
        )
        
        output = f"Installing dependencies from {requirements_file}...\n"
        output += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"stderr:\n{result.stderr}\n"
        
        if result.returncode == 0:
            output += "✅ Dependencies installed successfully!"
        else:
            output += f"❌ Installation failed with return code {result.returncode}"
        
        return output
        
    except Exception as e:
        return f"Error installing dependencies: {e}"

def run_tests(test_directory: str = "tests") -> str:
    """
    Runs Python tests in the specified directory.
    Args:
        test_directory: Directory containing test files (default: tests).
    Returns:
        Test execution results.
    """
    try:
        if not Path(test_directory).exists():
            return f"Error: Test directory {test_directory} not found."
        
        # Run pytest
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_directory, "-v"],
            capture_output=True,
            text=True
        )
        
        output = f"Running tests in {test_directory}...\n"
        output += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"stderr:\n{result.stderr}\n"
        
        if result.returncode == 0:
            output += "✅ All tests passed!"
        elif result.returncode == 1:
            output += "❌ Some tests failed."
        else:
            output += f"⚠️  Tests exited with return code {result.returncode}"
        
        return output
        
    except Exception as e:
        return f"Error running tests: {e}"

# ===== ADVANCED GIT INTEGRATION =====

def git_status() -> str:
    """
    Gets the current git status with detailed information about changes.
    Returns:
        Detailed git status information.
    """
    try:
        # Check if this is a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            return "Error: Not a git repository or git not installed."
        
        return f"Git Status:\n{result.stdout}"
        
    except Exception as e:
        return f"Error getting git status: {e}"

def git_commit_with_ai_message(files: str = ".") -> str:
    """
    Commits changes with an AI-generated commit message based on the changes.
    Args:
        files: Files to commit (default: "." for all changes).
    Returns:
        Commit result with the generated message.
    """
    try:
        # Get the diff to understand what changed
        diff_result = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True)
        if diff_result.returncode != 0:
            return "Error: No staged changes to commit."
        
        # Get unstaged changes if no staged changes
        if not diff_result.stdout.strip():
            diff_result = subprocess.run(['git', 'diff'], capture_output=True, text=True)
            if not diff_result.stdout.strip():
                return "No changes detected to commit."
        
        # Analyze the changes to generate a commit message
        changes = diff_result.stdout
        message = _generate_commit_message(changes)
        
        # Stage and commit
        subprocess.run(['git', 'add', files], check=True)
        commit_result = subprocess.run(['git', 'commit', '-m', message], capture_output=True, text=True)
        
        if commit_result.returncode == 0:
            return f"✅ Committed successfully!\nMessage: {message}\n\n{commit_result.stdout}"
        else:
            return f"❌ Commit failed: {commit_result.stderr}"
            
    except Exception as e:
        return f"Error during commit: {e}"

def _generate_commit_message(changes: str) -> str:
    """Generate a meaningful commit message based on the changes."""
    # Simple heuristic-based commit message generation
    lines_added = changes.count('+') - changes.count('+++')
    lines_removed = changes.count('-') - changes.count('---')
    
    if 'def ' in changes or 'class ' in changes:
        if 'test' in changes.lower():
            return "Add tests for new functionality"
        else:
            return "Add new functions and classes"
    elif 'import ' in changes:
        return "Update dependencies and imports"
    elif 'print(' in changes:
        return "Add debugging and logging"
    elif lines_added > lines_removed * 2:
        return "Add new features and functionality"
    elif lines_removed > lines_added * 2:
        return "Remove unused code and cleanup"
    else:
        return "Update and refactor existing code"

def git_resolve_conflicts() -> str:
    """
    Attempts to automatically resolve git merge conflicts.
    Returns:
        Status of conflict resolution.
    """
    try:
        # Check for conflicts
        status_result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if 'both modified' not in status_result.stdout:
            return "No merge conflicts detected."
        
        # Get list of conflicted files
        conflicted_files = []
        for line in status_result.stdout.split('\n'):
            if 'both modified' in line:
                file_path = line.split()[-1]
                conflicted_files.append(file_path)
        
        if not conflicted_files:
            return "No conflicted files found."
        
        output = ["🔄 Attempting to resolve conflicts in:"]
        for file_path in conflicted_files:
            output.append(f"   📁 {file_path}")
            
            # Try to resolve conflicts automatically
            try:
                # Read the conflicted file
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Simple conflict resolution: prefer current branch
                resolved_content = re.sub(r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> .*\n', r'\1', content, flags=re.DOTALL)
                
                # Write resolved content
                with open(file_path, 'w') as f:
                    f.write(resolved_content)
                
                # Stage the resolved file
                subprocess.run(['git', 'add', file_path], check=True)
                output.append(f"      ✅ Resolved automatically")
                
            except Exception as e:
                output.append(f"      ❌ Could not resolve: {e}")
        
        output.append("\n💡 Note: Automatic resolution may not always be correct.")
        output.append("   Review the changes before committing.")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error resolving conflicts: {e}"

def git_smart_branch(operation: str, branch_name: str = None) -> str:
    """
    Performs intelligent git branch operations with safety checks.
    Args:
        operation: Branch operation ('create', 'switch', 'merge', 'delete').
        branch_name: Name of the branch (required for most operations).
    Returns:
        Result of the branch operation.
    """
    try:
        if operation == 'create':
            if not branch_name:
                return "Error: Branch name required for create operation."
            
            # Create and switch to new branch
            subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
            return f"✅ Created and switched to branch '{branch_name}'"
            
        elif operation == 'switch':
            if not branch_name:
                return "Error: Branch name required for switch operation."
            
            # Check if branch exists
            result = subprocess.run(['git', 'branch', '--list', branch_name], capture_output=True, text=True)
            if not result.stdout.strip():
                return f"❌ Branch '{branch_name}' does not exist."
            
            # Switch to branch
            subprocess.run(['git', 'checkout', branch_name], check=True)
            return f"✅ Switched to branch '{branch_name}'"
            
        elif operation == 'merge':
            if not branch_name:
                return "Error: Branch name required for merge operation."
            
            # Check current branch
            current = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
            if current == branch_name:
                return "❌ Cannot merge branch into itself."
            
            # Merge the branch
            result = subprocess.run(['git', 'merge', branch_name], capture_output=True, text=True)
            if result.returncode == 0:
                return f"✅ Successfully merged '{branch_name}' into '{current}'"
            else:
                return f"❌ Merge failed: {result.stderr}"
                
        elif operation == 'delete':
            if not branch_name:
                return "Error: Branch name required for delete operation."
            
            # Check if we're trying to delete current branch
            current = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
            if current == branch_name:
                return "❌ Cannot delete current branch. Switch to another branch first."
            
            # Delete the branch
            subprocess.run(['git', 'branch', '-d', branch_name], check=True)
            return f"✅ Deleted branch '{branch_name}'"
            
        else:
            return f"Error: Unknown operation '{operation}'. Valid operations: create, switch, merge, delete"
            
    except Exception as e:
        return f"Error during branch operation: {e}"

# ===== REAL-TIME CODE MONITORING =====

def monitor_code_quality_continuous() -> str:
    """
    Continuously monitors code quality and provides real-time feedback.
    Returns:
        Current code quality status and recommendations.
    """
    try:
        output = ["Real-time Code Quality Monitor"]
        output.append("=" * 50)
        
        # Get all Python files
        python_files = list(Path(".").rglob("*.py"))
        output.append(f"Monitoring {len(python_files)} Python files")
        
        # Quick quality scan
        issues = []
        for py_file in python_files[:20]:  # Limit to first 20 for performance
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for common issues
                if 'print(' in content:
                    issues.append(f"{py_file}: Contains print statements")
                if len(content.split('\n')) > 500:
                    issues.append(f"{py_file}: Very long file (>500 lines)")
                if content.count('TODO') > 5:
                    issues.append(f"{py_file}: Many TODO items")
                    
            except Exception:
                continue
        
        if issues:
            output.append("\nIssues Found:")
            for issue in issues[:10]:  # Show first 10 issues
                output.append(f"   {issue}")
        else:
            output.append("\nNo immediate issues detected")
        
        output.append("\nRecommendations:")
        output.append("   - Run 'python -m gcode.cli \"fix code quality issues\"' for auto-fixes")
        output.append("   - Use 'context' command to see detailed project insights")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error monitoring code quality: {e}"

def auto_fix_common_issues() -> str:
    """
    Automatically fixes common code quality issues.
    Returns:
        Report of fixes applied.
    """
    try:
        output = ["Auto-fixing Common Issues"]
        output.append("=" * 50)
        
        fixes_applied = 0
        
        # Get Python files
        python_files = list(Path(".").rglob("*.py"))
        
        for py_file in python_files[:10]:  # Limit to first 10
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix 1: Remove trailing whitespace
                content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
                
                # Fix 2: Ensure single newline at end
                content = content.rstrip() + '\n'
                
                # Fix 3: Remove multiple blank lines
                content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
                
                # Fix 4: Basic import organization (simple version)
                lines = content.split('\n')
                import_lines = []
                other_lines = []
                
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                
                if import_lines:
                    # Sort imports
                    import_lines.sort()
                    content = '\n'.join(import_lines) + '\n\n' + '\n'.join(other_lines)
                
                # Write back if changed
                if content != original_content:
                    with open(py_file, 'w') as f:
                        f.write(content)
                    fixes_applied += 1
                    output.append(f"Fixed {py_file}")
                
            except Exception as e:
                output.append(f"Could not process {py_file}: {e}")
        
        output.append(f"\nTotal fixes applied: {fixes_applied}")
        output.append("Run 'git diff' to see what changed")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error auto-fixing issues: {e}"

# ===== ADVANCED TESTING =====

def generate_property_based_tests(source_file: str) -> str:
    """
    Generates property-based tests using hypothesis for a Python file.
    Args:
        source_file: The source Python file to generate tests for.
    Returns:
        Generated property-based test file.
    """
    try:
        # Read the source file
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Parse to find functions
        try:
            tree = ast.parse(content)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            return f"Error: Could not parse {source_file} - syntax error"
        
        if not functions:
            return f"No functions found in {source_file}"
        
        # Generate test file path
        source_path = Path(source_file)
        test_dir = source_path.parent / "tests"
        test_file = test_dir / f"test_property_{source_path.stem}.py"
        
        # Create test directory if it doesn't exist
        test_dir.mkdir(exist_ok=True)
        
        # Generate property-based test content
        test_content = f'''import pytest
from hypothesis import given, strategies as st
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from {source_path.stem} import *
except ImportError as e:
    print(f"Warning: Could not import from {{source_path.stem}}: {{e}}")

'''
        
        # Generate property-based tests for each function
        for func in functions:
            # Generate appropriate strategies based on function signature
            strategies = []
            for arg in func.args.args:
                if arg.arg == 'self':
                    continue
                # Simple strategy generation - can be enhanced
                strategies.append(f"st.text(min_size=1, max_size=10)")
            
            if strategies:
                test_content += f'''
@given({', '.join(strategies)})
def test_{func.name}_properties({', '.join([arg.arg for arg in func.args.args if arg.arg != 'self'])}):
    """Property-based test for {func.name} function."""
    try:
        # Test that function doesn't crash on valid inputs
        result = {func.name}({', '.join([arg.arg for arg in func.args.args if arg.arg != 'self'])})
        # Add more specific property checks here
        assert result is not None  # Basic property
    except Exception as e:
        # Function should handle errors gracefully
        assert isinstance(e, (ValueError, TypeError, AttributeError))
'''
        
        # Write the test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        return f"✅ Generated property-based tests at {test_file}\nGenerated tests for {len(functions)} functions using hypothesis."
        
    except Exception as e:
        return f"Error generating property-based tests: {e}"

def run_security_scan() -> str:
    """
    Runs a basic security scan on Python files for common vulnerabilities.
    Returns:
        Security scan results and recommendations.
    """
    try:
        output = ["Security Vulnerability Scan"]
        output.append("=" * 50)
        
        vulnerabilities = []
        python_files = list(Path(".").rglob("*.py"))
        
        for py_file in python_files[:50]:  # Limit for performance
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for common security issues
                if 'eval(' in content:
                    vulnerabilities.append(f"{py_file}: Uses eval() - potential code injection")
                if 'exec(' in content:
                    vulnerabilities.append(f"{py_file}: Uses exec() - potential code injection")
                if 'subprocess.run(' in content and 'shell=True' in content:
                    vulnerabilities.append(f"{py_file}: Uses shell=True - potential command injection")
                if 'pickle.loads(' in content:
                    vulnerabilities.append(f"{py_file}: Uses pickle.loads() - potential deserialization attack")
                if 'input(' in content:
                    vulnerabilities.append(f"{py_file}: Uses input() - validate user input")
                if 'os.system(' in content:
                    vulnerabilities.append(f"{py_file}: Uses os.system() - use subprocess instead")
                    
            except Exception:
                continue
        
        if vulnerabilities:
            output.append("\nSecurity Issues Found:")
            for vuln in vulnerabilities[:15]:  # Show first 15
                output.append(f"   {vuln}")
        else:
            output.append("\nNo obvious security vulnerabilities detected")
        
        output.append("\nSecurity Recommendations:")
        output.append("   - Avoid eval(), exec(), and pickle.loads() with untrusted input")
        output.append("   - Use subprocess.run() with shell=False")
        output.append("   - Validate and sanitize all user inputs")
        output.append("   - Use secrets module for cryptographic operations")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error during security scan: {e}"

def performance_profiling(file_path: str = None) -> str:
    """
    Performs basic performance profiling on Python code.
    Args:
        file_path: Specific file to profile (default: None for project-wide).
    Returns:
        Performance analysis and recommendations.
    """
    try:
        output = ["Performance Profiling"]
        output.append("=" * 50)
        
        if file_path:
            files_to_check = [Path(file_path)]
        else:
            files_to_check = list(Path(".").rglob("*.py"))[:20]  # Limit for performance
        
        performance_issues = []
        
        for py_file in files_to_check:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for performance anti-patterns
                if content.count('for ') > content.count('while '):
                    # Check for nested loops
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'for ' in line:
                            # Look for nested loops
                            for j in range(i+1, min(i+10, len(lines))):
                                if 'for ' in lines[j] and lines[j].strip().startswith('for '):
                                    performance_issues.append(f"{py_file}: Potential O(n²) nested loops around line {i+1}")
                                    break
                
                if 'list(' in content and 'range(' in content:
                    performance_issues.append(f"{py_file}: Consider using generators instead of list(range())")
                
                if content.count('import *') > 0:
                    performance_issues.append(f"{py_file}: Wildcard imports can slow down startup")
                    
            except Exception:
                continue
        
        if performance_issues:
            output.append("\nPerformance Issues Found:")
            for issue in performance_issues[:10]:
                output.append(f"   {issue}")
        else:
            output.append("\nNo obvious performance issues detected")
        
        output.append("\nPerformance Tips:")
        output.append("   - Use generators instead of lists for large datasets")
        output.append("   - Avoid nested loops when possible")
        output.append("   - Use set() for membership testing")
        output.append("   - Profile with cProfile for detailed analysis")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error during performance profiling: {e}"
