# STS agent — common workflows
# Run `just` to see available recipes.

# Run tests (fast, no build)
test:
    python -m pytest tests/ -q

# Run tests with verbose output
test-v:
    python -m pytest tests/ -v

# Run a specific test file or pattern
# Usage: just run tests/test_random_agent.py
run *ARGS:
    python -m pytest {{ARGS}}
