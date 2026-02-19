# CLAUDE.md

This file provides guidance to Claude when working in this repository.

## Project Overview

Python backend/API project.

## Development Setup

- Document how to install dependencies here (e.g., `pip install -r requirements.txt` or `uv sync`)
- Document how to run the server/app locally
- Document environment variable requirements (e.g., copy `.env.example` to `.env`)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run the application
python main.py
```

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes

## Architecture Notes

- Document key architectural decisions here
- List important modules and their responsibilities
- Note any non-obvious patterns or conventions used in this codebase

## Testing

- Tests live in `tests/`
- Run `pytest` to execute the test suite
- Write tests for all new functionality

## What to Avoid

- Do not commit secrets or credentials
- Do not modify lock files manually
- Do not break existing public interfaces without updating all callers
