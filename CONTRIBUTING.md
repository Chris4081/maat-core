# Contributing to MAAT-Core

Thank you for your interest in contributing to MAAT-Core! 🎉

This document provides guidelines for contributing to the project.

---

## Ways to Contribute

### 1. Report Issues 🐛

Found a bug or have a suggestion? Please open an issue!

**Before opening an issue:**
- Check if a similar issue already exists
- Provide a clear, descriptive title
- Include steps to reproduce (for bugs)
- Include your Python version and OS

**Good issue example:**
```
Title: Constraint margin returns NaN for edge case

Description:
When using a constraint with x=0 and division by x, 
the margin returns NaN instead of raising an error.

Steps to reproduce:
1. Define constraint: lambda s: 1.0 / s.val
2. Call with state where s.val = 0
3. Observe NaN in constraint report

Expected: Clear error message
Actual: Silent NaN

Python: 3.9.7
OS: Ubuntu 22.04
```

### 2. Suggest Features 💡

Have an idea for improvement? We'd love to hear it!

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Optional: implementation sketch

### 3. Improve Documentation 📚

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples
- Improve code comments
- Translate to other languages

### 4. Add Examples 🧪

New examples help others learn MAAT-Core:
- Real-world use cases
- Domain-specific applications
- Educational demos

### 5. Submit Code 💻

Code contributions via pull requests are welcome!

---

## Development Setup

### 1. Fork and clone

```bash
# Fork on GitHub, then:
git clone https://github.com/chris4081/maat-core.git
cd maat-core
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install in editable mode

```bash
pip install -e ".[dev]"
```

### 4. Create a branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

---

## Code Guidelines

### Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and small

**Example:**
```python
def evaluate_constraint(state, constraint):
    """
    Evaluate a constraint margin for a given state.
    
    Args:
        state: State object with relevant attributes
        constraint: Constraint object with margin function
        
    Returns:
        float: Constraint margin (positive = satisfied)
    """
    return constraint.margin(state)
```

### Testing

- Add tests for new features
- Ensure existing tests pass
- Test edge cases

```bash
# Run tests (when test suite exists)
pytest tests/
```

### Commits

- Write clear, descriptive commit messages
- Reference issue numbers when relevant

**Good commit messages:**
```
Add constraint margin visualization helper

Fix NaN handling in constraint evaluation (#42)

Improve documentation for Field class

Add healthcare allocation example
```

**Bad commit messages:**
```
fix bug
update code
changes
asdf
```

---

## Pull Request Process

### 1. Before submitting

- [ ] Code follows style guidelines
- [ ] Tests pass (if applicable)
- [ ] Documentation updated (if needed)
- [ ] Examples work (if added)
- [ ] Commit messages are clear

### 2. Submit PR

- Reference related issues
- Describe what changed and why
- Include examples if relevant

**PR template:**
```markdown
## Description
Brief description of changes

## Related Issues
Fixes #42

## Changes
- Added feature X
- Fixed bug Y
- Updated documentation

## Testing
- [ ] Tested on Python 3.8
- [ ] Tested on Python 3.11
- [ ] Added new tests

## Screenshots (if applicable)
[Add screenshots for visual changes]
```

### 3. Review process

- Maintainer will review your PR
- Address feedback and comments
- Once approved, PR will be merged

---

## Example Contributions

### Adding a new example

```bash
# Create new example file
touch examples/my_new_example.py

# Add to examples/README.md
# Update docs/DOCUMENTATION.md if relevant
```

### Fixing a bug

```bash
# Create fix branch
git checkout -b fix/constraint-nan-handling

# Make changes
# Add tests
# Commit

git commit -m "Fix NaN handling in constraint evaluation (#42)"

# Push and create PR
git push origin fix/constraint-nan-handling
```

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Trolling or inflammatory comments
- Publishing others' private information

### Enforcement

Report unacceptable behavior to: christof.krieg@outlook.de

---

## Questions?

- 💬 Ask in [GitHub Discussions](https://github.com/Chris4081/maat-core/discussions)
- 📧 Email: christof.krieg@outlook.de
- 🐛 Open an issue for bugs

---

## Recognition

Contributors will be acknowledged in:
- README.md acknowledgments section
- Release notes
- Citation if contribution is substantial

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Thank You! 🙏

Every contribution helps make MAAT-Core better for everyone.

We appreciate your time and effort!

**Happy contributing! 🚀**
