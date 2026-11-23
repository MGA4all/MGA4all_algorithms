# Submitting Contributions

This repository accepts contributions through pull requests. If you would like to contribute, create a fork (or a feature branch if you have write access), make your changes, and open a pull request for review.

Before opening a pull request, please ensure that:

- all tests pass;
- code has been formatted with the configured pre-commit hooks;
- any relevant documentation has been updated.

# Development Setup

MGA4all uses:

- **Pixi** for environment management;
- **Hatchling** for packaging and version management;
- **pip** for installing MGA4all itself into the development environment.

> **Note**
>
> - Some pytest options are enabled by default via the `tool.pytest.ini_options` section in `pyproject.toml`.
> - The ORACLE implementation depends on `cddlib` and `pycddlib`, which are installed as part of the development setup described below.

## Using Pixi (recommended)

Clone the repository and create the development environment:

```bash
git clone https://github.com/MGA4all/MGA4all.git
cd MGA4all

pixi install
```

Install MGA4all and additional Python dependencies:

```bash
pixi run bootstrap
```

Validate formatting and linting:

```bash
pixi run lint
```

Run the test suite:

```bash
pixi run test
```

## Managing Environments Manually

Advanced users may prefer to install MGA4all into an existing Python environment, for example an existing PyPSA-Eur environment.

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install MGA4all:

```bash
pip install -e .
```

Install additional dependencies required by the ORACLE implementation:

```bash
pip install pycddlib
```

You are also responsible for providing the required native libraries and solver dependencies, including:

- cddlib
- HiGHS
- Gurobi (requires a valid license)

Run tests:

```bash
pytest
```

# Pre-commit

This repository uses [pre-commit](https://www.pre-commit.com) to automatically perform formatting, linting, and validation checks on source code and configuration files.

With the Pixi development environment installed:

```bash
pixi shell pre-commit install
```

The hooks will then run automatically whenever you create a commit.

Useful commands:

```bash
pre-commit run
```

Run all configured hooks on files staged for commit.

```bash
pre-commit run --all-files
```

Run all hooks on the entire repository.

Alternatively, run the configured Pixi task:

```bash
pixi run lint
```

which executes:

```bash
pre-commit run --all-files
```

and is the recommended check before opening a pull request.
