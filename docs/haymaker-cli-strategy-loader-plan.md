# Haymaker CLI Strategy Loader Plan

## Status

Implemented as part of the runtime-context entrypoint refactor.

## Goal

Add a stable console entry point for live trading:

```bash
haymaker /home/tomek/ib_strategies/temp.py
```

Haymaker should become the process entry point and load external user strategy
code. Strategy files remain outside the Haymaker repository.

## Current Behaviour

User strategy files import `haymaker.app.App`, construct streamers and pipelines
at module scope, instantiate `App`, and call `App.run()` under:

```python
if __name__ == "__main__":
    ...
```

This allows a short command such as:

```bash
uv run temp.py
```

The new command should remain similarly concise. It must not require a config
file argument because Haymaker already supports YAML overrides, environment
variables, and command-line overrides.

## Implemented Behaviour

Add a `haymaker` console script that:

1. Parses a required positional strategy module path together with existing
   live-app configuration options.
2. Resolves the strategy path and inserts its parent directory into `sys.path`.
3. Loads the external file exactly once with `importlib`.
4. Allows module execution to register streamers, pipelines, portfolios, and
   strategy objects.
5. Creates `RuntimeContext`, binds strategy-module metadata, and runs the
   Haymaker application after loading succeeds.

The external strategy file should stop creating and running `App` itself. Data
that is strategy-owned, such as `no_future_roll_strategies`, should stay with
the strategy module and be passed to the app/runtime by the loader.

## Configuration Compatibility

Preserve existing configuration precedence:

1. Command-line options.
2. YAML override file selected from the command line.
3. YAML override file selected through environment configuration.
4. `HAYMAKER_` environment variables.
5. Package defaults.

Loading `.env` inside the strategy file is too late if configuration is created
before strategy import. Environment variables must be established before the
`haymaker` process starts.

## Implementation Outline

1. Add a dedicated CLI module that performs argument parsing before importing
   runtime services.
2. Add `haymaker = "haymaker.cli:main"` under `[project.scripts]`.
3. Implement strategy loading with `importlib.util.spec_from_file_location`.
4. Preserve sibling imports from the external strategy directory.
5. Move runtime construction into `haymaker.runtime.RuntimeContext`.
6. Update execution and configuration documentation.
7. Add tests for path resolution, one-time execution, sibling imports,
   configuration precedence, and useful failure messages.

## Out Of Scope

- Packaging external strategy files with Haymaker.
- Committing user strategies into the Haymaker repository.
- Introducing a plugin framework.
