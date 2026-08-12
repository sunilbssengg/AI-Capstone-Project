# `env/` — Environment Setup

This folder documents how to set up an isolated Python environment for this
project. It intentionally does **not** contain the virtual environment
binaries themselves (those must never be committed to GitHub) — instead it
documents the two supported ways to create one, plus where runtime secrets
(`.env`) live.

## 1. Create a virtual environment named `env`

```bash
# from the project root
python -m venv env

# activate it
source env/bin/activate        # Linux / macOS
env\Scripts\activate           # Windows

# install dependencies
pip install -r requirements.txt
```

The folder `env/` (the actual virtual environment, once created by the
command above) is excluded via `.gitignore` — only this `README.md` is
tracked in GitHub so teammates know how to reproduce it.

## 2. Configure secrets (`.env`)

Runtime configuration (API keys, chunk sizes, vector DB paths, etc.) lives in
a `.env` file at the **project root**, not inside `env/`. This keeps the
Python virtual environment and the runtime configuration cleanly separated.

```bash
cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY (Gemini)
```

`core/config.py` loads this file automatically via `python-dotenv` at
application startup.

## 3. Verify

```bash
python -c "from core.config import settings; print(settings.GEMINI_LLM_MODEL)"
```

If this prints the model name without error, your environment and `.env`
file are both configured correctly.
