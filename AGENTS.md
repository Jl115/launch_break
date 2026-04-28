# Project: Lunch Attendance Prediction System

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: `uv` (managed via `pyproject.toml`)
- **Architecture**: Feature-Sliced Design (FSD)
- **Dependencies**: Typer, Rich, Pydantic, Requests, Pandas, NumPy, scikit-learn, joblib, holidays, Textual, textual-plotext

## Architecture Rules
1. **Strict dependency hierarchy**: `app` -> `features` -> `entities` -> `shared`
2. **No circular imports**: `shared` must never import from `entities`, `features`, or `app`.
3. **Encapsulation**: Features must not leak internal state. Use DTOs/Interfaces.
4. **OOP Style**: All services implement an ABC Interface.
5. **CLI is the thin layer**: All business logic lives in `features`.

## Code Style
- PEP8 compliance (max line length = 100)
- Type hints required for all public methods
- Docstrings for all classes and public methods (Google format)
- `__all__` must be defined in all package `__init__.py` files

## OCR Configuration
- **Primary Model**: `deepseek-ocr:3b` via Ollama (`localhost:11434`)
- **Prompt Format**: Must use `Extract the text in the image.\n<|grounding|>Given the layout of the image.`
- **Fallback Model**: `qwen3-vl:235b-cloud` via Ollama
- **Output Cleaning**: Strip all ANSI escape codes, HTML tags (`<|ref|>`, `<|det|>`), and TTY noise.
- **Raw Text Strategy**: If JSON generation fails, the `RawTextParser` uses regex heuristics to extract names and numbers from OCR text.

## Model Configuration
- **Primary Algorithm**: `sklearn.ensemble.RandomForestRegressor` in `MultiOutputRegressor`
- **Seasonality**: Temporal features (month, week_of_year, day_of_week) + Swiss (BE) holidays
- **Data Requirement**: Minimum 20 weeks of historical menu sheets for reliable training.
- **Output Categories**: `Erw`, `Ki`, `MA`, `MA-Ki` (Adults, Kids, Mitarbeiter, Mitarbeiter-Kids)

## Data Paths
- **Images**: `data/images/`
- **JSONs**: `data/json/`
- **Models**: `models/`
