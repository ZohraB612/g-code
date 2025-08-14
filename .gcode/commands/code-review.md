# Code Review

Automatically review code for quality, security, and best practices.

## Usage
```
gcode code-review [file_or_directory]
```

## What it does
1. Analyzes code for potential issues
2. Checks for security vulnerabilities
3. Suggests performance improvements
4. Identifies code smells and anti-patterns
5. Generates a comprehensive review report

## Examples
```bash
# Review current file
gcode code-review

# Review specific file
gcode code-review src/main.py

# Review entire directory
gcode code-review src/

# Review with specific focus
gcode code-review --focus security src/
gcode code-review --focus performance src/
gcode code-review --focus quality src/
```

## Output
The command generates a detailed report with:
- Code quality score
- Security findings
- Performance recommendations
- Best practice suggestions
- Specific code improvements

## Configuration
Customize review settings in `.gcode/config.json`:
```json
{
  "code_review": {
    "security_level": "strict",
    "quality_threshold": 0.8,
    "ignore_patterns": ["*.test.py", "legacy/*"]
  }
}
```
