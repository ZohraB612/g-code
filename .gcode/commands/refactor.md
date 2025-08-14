# Refactor Code

Automatically refactor code for better structure, performance, and maintainability.

## Usage
```
gcode refactor [file_or_directory] [--target quality|performance|structure]
```

## What it does
1. Analyzes code structure and complexity
2. Identifies refactoring opportunities
3. Suggests specific improvements
4. Optionally applies refactoring automatically
5. Generates before/after comparison

## Examples
```bash
# Refactor current file for quality
gcode refactor --target quality

# Refactor specific file for performance
gcode refactor src/algorithm.py --target performance

# Refactor entire module for structure
gcode refactor src/utils/ --target structure

# Preview refactoring without applying
gcode refactor --preview src/main.py

# Apply refactoring automatically
gcode refactor --apply src/main.py
```

## Refactoring Types

### Quality Improvements
- Extract long functions into smaller ones
- Remove code duplication
- Improve variable naming
- Add type hints and documentation

### Performance Optimizations
- Optimize loops and data structures
- Reduce memory usage
- Improve algorithm efficiency
- Cache expensive operations

### Structural Improvements
- Reorganize class hierarchies
- Split large modules
- Improve package structure
- Standardize coding patterns

## Configuration
Set refactoring preferences in `.gcode/config.json`:
```json
{
  "refactor": {
    "auto_apply": false,
    "backup_files": true,
    "max_function_length": 20,
    "complexity_threshold": 10
  }
}
```
