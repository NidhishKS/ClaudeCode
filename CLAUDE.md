# CLAUDE.md

This file provides guidance for AI assistants working with this repository.

## Repository Overview

**Repository:** ClaudeCode
**Status:** Newly initialized — no application code, build system, or tests have been added yet.

## Current State

This is an empty Git repository. There are no:
- Source files or application code
- Package manager configurations (package.json, requirements.txt, etc.)
- Build or test tooling
- CI/CD pipelines
- Linting or formatting configurations

## Getting Started

When setting up this project, establish the following:

1. **Project scaffolding** — Choose a language/framework and initialize with the appropriate tooling (e.g., `npm init`, `cargo init`, `go mod init`)
2. **Directory structure** — Create a clear source layout (`src/`, `tests/`, `docs/`, etc.)
3. **Build configuration** — Set up build scripts, compiler/transpiler configs
4. **Testing** — Add a test framework and write initial tests
5. **Linting/formatting** — Configure code quality tools
6. **CI/CD** — Add workflow definitions for automated checks
7. **Documentation** — Create README.md with project description, setup instructions, and usage

## Conventions for AI Assistants

### General Rules

- Read existing code before proposing changes — never modify files you haven't read
- Keep changes minimal and focused on the task at hand
- Do not add features, refactoring, or "improvements" beyond what is requested
- Avoid introducing security vulnerabilities (command injection, XSS, SQL injection, etc.)
- Prefer editing existing files over creating new ones

### Git Workflow

- Write clear, descriptive commit messages summarizing the "why" not just the "what"
- Commit only the files relevant to the change
- Do not commit secrets, credentials, or `.env` files

### Code Quality

- Follow the conventions already established in the codebase
- Add tests for new functionality
- Run existing tests and linters before committing
- Keep functions small and focused

## Updating This File

Update this CLAUDE.md whenever:
- The project structure changes significantly
- New tooling or frameworks are added
- Build, test, or deployment workflows are established
- Important conventions or patterns are adopted
