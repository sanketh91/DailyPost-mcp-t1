# Contributing to DailyPost MCP Server

Thank you for your interest in contributing! This document outlines our workflow to ensure smooth collaboration and safe deployments.

## Workflow Overview

We use a Git-based workflow with GitHub Actions for continuous integration and Cloud Run for deployment:

1. **Feature branches** → **Pull Requests** → **Code Review** → **Merge to main** → **Auto-deploy to Cloud Run**

All work must go through pull requests; direct pushes to `main` are not permitted.

## Branch Naming Convention

When creating a feature branch, use descriptive names following this pattern:

- `feature/<short-description>` for new features
  - Example: `feature/add-weaviate-search`
- `bugfix/<short-description>` for bug fixes
  - Example: `bugfix/timeout-error-handling`
- `docs/<short-description>` for documentation
  - Example: `docs/api-endpoints`

## Development Workflow

### Step 1: Pull Latest Code
```bash
git checkout main
git pull origin main
```

### Step 2: Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### Step 3: Make Changes and Commit
- Write focused, small commits with clear messages
- Test your changes locally before pushing
```bash
git add <files>
git commit -m "Clear description of changes"
```

### Step 4: Push and Create a Pull Request
```bash
git push origin feature/your-feature-name
```

Then go to GitHub and open a pull request into `main`.

## Pull Request Guidelines

Every PR must include:

1. **Clear Title**: Describe the feature or fix in a few words
2. **Description**: Explain what you changed and why
3. **Testing**: Document how you tested the changes (commands run, results)
4. **Impact**: Note any changes to environment variables, configuration, or dependencies

Example PR description:
```
## Changes
- Added Weaviate semantic search integration
- Refactored tool registry for dynamic loading

## Testing
Run: `pytest tests/test_weaviate.py -v`
Result: All tests passed (8/8)

## Notes
- Requires WEAVIATE_URL environment variable
- Backwards compatible with existing tools
```

## Testing Locally

Before pushing your changes, run tests:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linting (if configured)
python -m ruff check .
```

## Review and Merge

- A maintainer will review your PR
- Address any feedback or requested changes
- Once approved and all checks pass, the PR will be merged to `main`
- Cloud Run will automatically deploy the updated code

## Important Notes

- **Do not commit sensitive data** (API keys, secrets, tokens)
- Use environment variables for configuration
- Keep commits small and focused
- Write descriptive commit messages
- Always run tests before pushing

## Questions?

Reach out to the maintainers if you have questions about the workflow or need help.
