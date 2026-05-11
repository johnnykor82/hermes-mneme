# Development Guide

This document explains how to set up the development environment, run tests, and contribute to the Hermes Context Engine Plugin.

## Setting Up

1.  **Clone the repository** (if you haven't):
    ```bash
    git clone https://github.com/johnnykor82/hermes-mneme.git
    cd hermes-mneme
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs `pytest`, `ruff`, and other dev tools.

4.  **Install runtime dependencies**:
    ```bash
    pip install sqlite-vec tiktoken numpy requests
    ```

## Running Tests

We use `pytest` for testing. Tests are located in `tests/` directory.

- **Run all tests**:
    ```bash
    pytest tests/ -v
    ```

- **Run unit tests only**:
    ```bash
    pytest tests/unit/ -v
    ```

- **Run integration tests only**:
    ```bash
    pytest tests/integration/ -v
    ```

## Code Style

We use `ruff` for linting and formatting.

- **Check code**:
    ```bash
    ruff check .
    ```

- **Format code**:
    ```bash
    ruff format .
    ```

## Contributing

1.  **Fork the repository**.
2.  **Create a feature branch** (`git checkout -b feature/my-feature`).
3.  **Commit your changes** (`git commit -am 'Add some feature'`).
4.  **Push to the branch** (`git push origin feature/my-feature`).
5.  **Create a Pull Request**.

## Plugin Structure

```
plugins/context_engine/custom_router/
├── __init__.py
├── engine.py          # Main ContextEngine implementation
├── store.py           # SQLite Event Store
├── index.py           # Embedding Index (Jina + sqlite-vec)
├── classifier.py      # Intent Classifier
├── router.py          # Context Router
├── prompt_builder.py  # Prompt Assembly
├── observability.py   # Tracing & Metrics
├── tools.py           # Agent Memory Tools
├── segmenter.py       # Session Segmenter
├── graph.py           # Execution Graph
├── config.py          # Configuration Loader
└── tests/
    ├── unit/
    └── integration/
```

## Debugging

- Check Hermes logs: `~/.hermes/logs/agent.log`.
- Check plugin trace: `~/.hermes/plugins/custom_router/trace.jsonl`.
- Run Hermes with `--log-level DEBUG` for more details.
