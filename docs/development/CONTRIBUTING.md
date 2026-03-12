# Contributing to ML Threshold Selection

We welcome contributions to ML Threshold Selection! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/VincentDD1125/ML-Threshold-Selection/issues) page
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Attach relevant data files (if possible) and error messages

### Suggesting Enhancements

- Use the [GitHub Discussions](https://github.com/VincentDD1125/ML-Threshold-Selection/discussions) page
- Describe the enhancement clearly
- Explain why it would be useful
- Provide examples if applicable

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding standards below
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request**

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/
```

### Documentation

- Write clear docstrings for all functions and classes
- Use Google-style docstrings
- Update README.md for user-facing changes
- Add examples for new features

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/ml_threshold_selection --cov-report=html
```

## Development Setup

### Prerequisites

- Python 3.7 or higher
- Git

### Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ML-Threshold-Selection.git
cd ML-Threshold-Selection

# Add upstream remote
git remote add upstream https://github.com/VincentDD1125/ML-Threshold-Selection.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Switch to main branch
git checkout main

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project's coding standards
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Commit messages are clear and descriptive

### Pull Request Template

When creating a pull request, please include:

1. **Description**: What changes were made and why
2. **Type of Change**: Bug fix, new feature, documentation, etc.
3. **Testing**: How the changes were tested
4. **Breaking Changes**: Any breaking changes and migration steps
5. **Checklist**: Confirm all requirements are met

### Review Process

- All pull requests require review
- Address feedback promptly
- Keep pull requests focused and reasonably sized
- Update your pull request if requested

## Project Structure

```
src/ml_threshold_selection/
├── app_controller.py          # GUI controller
├── ui_layout.py              # Tkinter interface
├── ui_visualization.py       # Plotting utilities
├── data_io.py                # Data loading/saving
├── training_pipeline.py      # Training workflow
├── prediction_analysis.py    # Prediction analysis
├── export_results.py         # Results export
├── labeling.py               # Label generation
├── feature_utils.py          # Feature utilities
├── fabric_thresholds.py      # Threshold grid helpers
├── fabric_bootstrap.py       # Bootstrap analysis
├── fabric_logging.py         # Logging utilities
└── fabric_pipeline.py        # Fabric analysis pipeline
```

## Areas for Contribution

### High Priority

- **Performance Optimization**: Improve speed and memory usage
- **Additional ML Models**: Support for more classifiers
- **Enhanced Visualization**: Better plotting and analysis tools
- **Documentation**: Improve user guides and API documentation

### Medium Priority

- **Testing**: Increase test coverage
- **Error Handling**: Better error messages and recovery
- **Configuration**: More flexible configuration options
- **Export Formats**: Support for additional output formats

### Low Priority

- **GUI Improvements**: Enhanced user interface
- **Batch Processing**: Better support for large datasets
- **Integration**: Integration with other scientific tools

## Questions?

If you have questions about contributing, please:

- Open a [GitHub Discussion](https://github.com/VincentDD1125/ML-Threshold-Selection/discussions)
- Contact the maintainers
- Check existing issues and discussions

## License

By contributing to ML Threshold Selection, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ML Threshold Selection! 🎉
