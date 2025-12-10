## Loops

Loops is a repository that explores area calculation of loops 

### Files
- `generate_fake_loop.py` is used to generate fake data


### Usage
Use 
```sh
uv run filename.py
```
to run any file 

## Setup 

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It automatically creates and manages a virtual environment for you. To use UV:

0. If you haven't, install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [installation instructions](https://github.com/astral-sh/uv#installation))
1. Install dependencies: `uv sync`
2. `cd` into the correct folder `cd YYYY/day_x/`
3. Run code: `uv run python3 day_x.py` 
4. Run tests: `uv run pytest -vv` 
