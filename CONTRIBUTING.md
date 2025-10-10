# Contributing to Historia Scribe

Thank you for your interest in contributing to Historia Scribe! We welcome contributions from the community and are excited to work with you.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Questions and Help](#questions-and-help)

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Exercise consideration and respect in your speech and actions
- Attempt collaboration before conflict
- Refrain from demeaning, discriminatory, or harassing behavior
- Be mindful of your surroundings and fellow participants

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/historia-scribe.git
   cd historia-scribe
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ryan-tris-walmsley/historia-scribe.git
   ```

4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- CUDA-capable GPU (recommended for training)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .
```

### Testing the Setup
```bash
python test_functionality.py
```

## Making Changes

### Project Structure
```
historia-scribe/
├── src/                    # Source code
│   ├── data/              # Data management
│   ├── preprocess/        # Image preprocessing
│   ├── model/            # Model training and inference
│   ├── evaluate/         # Model evaluation
│   └── app/              # GUI application
├── configs/              # Configuration files
├── docs/                 # Documentation
└── tests/                # Unit tests
```

### Areas for Contribution

1. **Core Features**:
   - Improve HTR accuracy
   - Add new preprocessing techniques
   - Enhance GUI functionality
   - Add support for new languages

2. **Documentation**:
   - Tutorials and guides
   - API documentation
   - User manuals

3. **Testing**:
   - Unit tests
   - Integration tests
   - Performance benchmarks

4. **Infrastructure**:
   - CI/CD improvements
   - Docker support
   - Packaging improvements

## Pull Request Process

1. **Keep it focused**: Each PR should address a single issue or feature
2. **Write tests**: Include tests for new functionality
3. **Update documentation**: Update relevant documentation
4. **Follow code style**: Ensure your code follows our style guidelines
5. **Squash commits**: Clean up your commit history before submitting

### PR Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] PR description explains the changes

## Code Style

We use several tools to maintain code quality:

### Formatting
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/
```

### Linting
```bash
# Run pylint
pylint src/

# Type checking (if applicable)
mypy src/
```

### Code Style Guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions focused and small

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests
- Write unit tests for individual functions
- Write integration tests for workflows
- Mock external dependencies
- Test edge cases and error conditions

## Documentation

### Building Documentation
```bash
cd docs
make html
```

### Documentation Structure
- User guides in `docs/user_guide/`
- API reference in `docs/api/`
- Development guides in `docs/development/`

## Questions and Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact ryan.tris.walmsley@gmail.com for direct questions

## Recognition

Contributors will be recognized in:
- The project's README.md
- Release notes
- The contributors graph on GitHub

Thank you for contributing to Historia Scribe! Your efforts help make historical documents more accessible to researchers worldwide.
