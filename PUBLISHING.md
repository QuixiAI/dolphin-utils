# Publishing to PyPI

This guide explains how to publish the `dolphin-utils` package to PyPI.

## Prerequisites

1. Install build and upload tools:
```bash
pip install build twine
```

2. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

## Building the Package

Build the distribution files:
```bash
python -m build
```

This creates:
- `dist/dolphin_utils-X.X.X.tar.gz` (source distribution)
- `dist/dolphin_utils-X.X.X-py3-none-any.whl` (wheel distribution)

## Testing on TestPyPI

1. Upload to TestPyPI first:
```bash
python -m twine upload --repository testpypi dist/*
```

2. Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ dolphin-utils
```

3. Test the CLI command:
```bash
dolphin-summarize --help
```

## Publishing to PyPI

Once testing is successful:

1. Upload to PyPI:
```bash
python -m twine upload dist/*
```

2. Verify installation:
```bash
pip install dolphin-utils
dolphin-summarize --help
```

## Version Management

Update the version in `pyproject.toml` before each release:
```toml
[project]
name = "dolphin-utils"
version = "0.5.0"  # Increment this
```

## Package Contents

The package provides:
- **Package name**: `dolphin-utils`
- **CLI command**: `dolphin-summarize`
- **Python module**: `python -m dolphin_summarize`
- **Import**: `import dolphin_summarize`

## Notes

- The package name is `dolphin-utils` (with hyphen)
- The Python module is `dolphin_summarize` (with underscore)
- The CLI command is `dolphin-summarize` (with hyphen)
- This follows Python packaging conventions
