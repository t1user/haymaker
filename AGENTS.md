# General Coding Rules

- Prefer minimal, surgical changes.
- Do not rewrite unrelated code.
- Preserve existing architecture unless instructed otherwise.
- Explain tradeoffs before major refactors.
- Prefer readability over cleverness.
- Avoid adding dependencies unless justified.
- Run tests after changes when possible.
- If tests fail, explain whether the failure is related to your changes.
- Keep functions focused and short.
- Avoid unnecessary abstraction.
- Prefer explicit code over metaprogramming.

# Workflow

- Before editing, briefly explain understanding of the task.
- For non-trivial tasks, create a short implementation plan.
- After changes, summarize modified files and rationale.
- Update relevant documentation when architecture, setup, commands, conventions,
  workflows, or important assumptions change.
- For research code under `haymaker/research`, read
  `haymaker/research/AGENTS.md` before editing.
- Avoid importing `haymaker.app` in tests unless explicitly needed. It sets up
  logging and imports runtime singletons; prefer lower-level modules for
  focused tests.
- Create docstrings for any new functions/classes/methods
- Use google-style, sphinx compatible docstrings
- If changing user-relevant behaviour, scan existing documentation and update any sections relevant to this newly changed behaviour

# Git

- Never commit unless explicitly instructed.
- When committing, add `[llm]` at the end of the commit message.
- Before committing, make sure any file with secrets is included in `.gitignore`.
- Never push.
- Do not modify `.env` files.
- Do not delete files without explanation.
- Run `mypy` and `pytest`, do not commit if there are any issues.

# Python Preferences

- Prefer type hints.
- Prefer `pathlib` over `os.path`.
- Use `pytest` to create tests.
- Use `black` for formatting.
- Add docstrings for new functions and classes.

# Validation

- Run the narrowest meaningful test command for the changed area first.
- For broad Python changes, run:

```bash
python -m pytest
```

- For focused research changes, run:

```bash
python -m pytest tests/test_research
```

- For typing and lint checks when touching research code, run when practical:

```bash
python -m mypy haymaker/research tests/test_research
python -m flake8 haymaker/research tests/test_research --select=F401,F821,F841,E501
```

# Project Notes

- This is an Interactive Brokers trading framework built around `ib_insync`,
  event-driven runtime components, a historical dataloader, dataframe-first
  research tools, and a local Streamlit dashboard.
- Live trading, controller sync, futures rolling, and order reconciliation are
  high-risk areas. Keep changes especially narrow and well verified there.
- Configuration is primarily YAML-driven. Do not change real local `.env` files
  or credential files.
- See `docs/codebase-map.md` for the current repository map.
