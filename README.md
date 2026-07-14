# Auto SANE

The automatic scanning Tool

# Install

For setting up this application in your python environment just run

```bash
pip install git+https://github.com/gaggenwaschke/auto_sane
```

```bash
uv add git+https://github.com/gaggenwaschke/auto_sane
```

# Development

For setting this up for development run

```bash
git clone https://github.com/gaggenwaschke/auto_sane
cd auto_sane
uv sync --all-extras  # installs all dependencies, including the development ones
uv run pre-commit install  # installs all linters as pre commit hooks
```
