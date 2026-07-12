# Auto SANE

The automatic scanning Tool

# Install

For setting up this application in your python environment just run

```bash
pip install git+<URL>
```

```bash
uv add git+<URL>
```

# Development

For setting this up for development run

```bash
git clone <URL>
cd auto_sane
uv sync --all-extras  # installs all dependencies, including the development ones
uv run pre-commit install  # installs all linters as pre commit hooks
```
