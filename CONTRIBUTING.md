# Contributing

Thank you for your interest in contributing to the Telegram RAG Chatbot! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose
- Git

### Local Environment

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd telegram-rag-chatbot
   ```

2. **Set up the telegram-bot environment:**
   ```bash
   cd telegram-bot
   uv sync --dev
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows
   .\.venv\Scripts\Activate.ps1

   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Start infrastructure services:**
   ```bash
   docker-compose up qdrant ml-api -d
   ```

5. **Configure environment:**
   ```bash
   cp ../.env.example ../.env
   # Edit .env with your credentials
   ```

## Code Quality Standards

This project maintains strict code quality standards. All contributions must pass the following checks:

### Linting

```bash
ruff check app/
```

### Formatting

```bash
ruff format app/
```

### Type Checking

```bash
mypy app/
```

### Testing

```bash
pytest tests/ -v --cov=app
```

**Minimum coverage:** 80%

## Pre-Commit Hooks

Install pre-commit hooks to automatically check code before commits:

```bash
pre-commit install
```

Run all hooks manually:

```bash
pre-commit run --all-files
```

## Commit Guidelines

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New features
- `fix` - Bug fixes
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or modifying tests
- `chore` - Maintenance tasks

**Examples:**
```
feat(handlers): add /summarize command for document summaries
fix(cache): resolve LRU eviction race condition
docs(readme): update bot commands table
test(orchestrator): add integration tests for hybrid search
```

### Branch Naming

Use descriptive branch names:

```
feat/add-summarize-command
fix/cache-eviction-bug
docs/update-architecture-diagrams
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following code quality standards

3. **Run all checks:**
   ```bash
   ruff check app/
   ruff format app/
   mypy app/
   pytest tests/ -v
   ```

4. **Commit with a descriptive message**

5. **Push and create Pull Request:**
   ```bash
   git push origin feat/your-feature-name
   ```

6. **Fill out the PR template** with:
   - Description of changes
   - Related issue (if any)
   - Testing performed
   - Screenshots (for UI changes)

## Project Structure

```
telegram-bot/
├── app/
│   ├── handlers/       # Command handlers
│   ├── rag/            # RAG pipeline
│   ├── chunking/       # Document chunking
│   ├── database/       # Qdrant client
│   ├── models/         # LLM providers
│   ├── services/       # External API clients
│   └── utils/          # Utilities
├── tests/              # Test suite
└── sample_docs/        # Sample documents

ml-api/
├── ml_api.py           # FastAPI service
└── Dockerfile
```

## Adding New Features

### Adding a New Command

1. Create handler function in `app/handlers/`:
   ```python
   async def new_command(
       update: Update,
       context: ContextTypes.DEFAULT_TYPE,
       # ... dependencies
   ) -> None:
       """
       Handle /newcommand.

       Args:
           update: Telegram update object
           context: Bot context
       """
       # Implementation
   ```

2. Register in `app/main.py`:
   ```python
   async def new_handler(update, context):
       await new_command(update, context, self.dependency)

   app.add_handler(CommandHandler("newcommand", new_handler))
   ```

3. Update `/help` text in `app/handlers/help.py`

4. Add tests in `tests/test_handlers.py`

5. Update documentation:
   - `README.md` - Bot Commands table
   - `CHANGELOG.md` - Under [Unreleased]

### Adding a New ML Endpoint

1. Add endpoint in `ml-api/ml_api.py`
2. Add client method in `app/services/ml_api_client.py`
3. Add tests for both
4. Update `ARCHITECTURE.md` - API Reference section

## Testing Guidelines

### Test Structure

```python
@pytest.mark.asyncio
async def test_feature_description():
    """Test that feature behaves correctly under condition."""
    # Arrange
    input_data = prepare_test_data()

    # Act
    result = await function_under_test(input_data)

    # Assert
    assert result == expected_value
```

### Mocking External Services

```python
@pytest.fixture
def mock_ml_client(mocker):
    """Mock ML API client for unit tests."""
    mock = mocker.patch("app.services.ml_api_client.MLAPIClient")
    mock.embed.return_value = {"dense_vecs": [...], "sparse_vecs": [...]}
    return mock
```

## Documentation

All code changes should include corresponding documentation updates:

- **Docstrings:** All public functions and classes
- **README.md:** User-facing features
- **ARCHITECTURE.md:** Technical changes
- **CHANGELOG.md:** Notable changes

## Getting Help

- Open an issue for bug reports or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

Thank you for contributing!
