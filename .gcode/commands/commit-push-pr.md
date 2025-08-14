# Commit, Push, and Create PR

Automatically commit changes, push to remote, and create a pull request.

## Usage
```
gcode commit-push-pr
```

## What it does
1. Analyzes git status and staged changes
2. Generates a descriptive commit message using AI
3. Commits the changes
4. Pushes to the current branch
5. Creates a pull request with AI-generated description

## Requirements
- Git repository with remote origin
- Staged changes to commit
- GitHub CLI (gh) installed and authenticated

## Examples
```bash
# Stage all changes and create PR
git add .
gcode commit-push-pr

# Stage specific files and create PR
git add src/ tests/
gcode commit-push-pr
```

## Configuration
Set your preferred branch naming convention in `.gcode/config.json`:
```json
{
  "git": {
    "branch_prefix": "feature/",
    "commit_template": "conventional"
  }
}
```
