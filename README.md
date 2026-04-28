# Lunch Attendance Prediction System

A Python CLI application that uses **OCR** (via Ollama) and **Machine Learning** (Random Forest Regressor) to predict weekday lunch attendance from scanned signup/menu sheets.

## What It Does

The system takes scanned weekly menu sheets (images), extracts structured data using AI-powered OCR, and trains an ML model to predict future lunch attendance across four categories:

- **Erw** – Adults  
- **Ki** – Kids  
- **MA** – Mitarbeiter (Staff)  
- **MA-Ki** – Staff Kids

The prediction aggregates per weekday (Mon–Fri) for a target month.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Python package manager)
- **[Ollama](https://ollama.com)** running locally with one of these models:
  - Primary: `deepseek-ocr:3b`
  - Fallback: `qwen3-vl:235b-cloud` (used automatically if primary fails)

### Install Ollama Models

```bash
ollama pull deepseek-ocr:3b
ollama pull qwen3-vl:235b-cloud   # recommended fallback
```

> Make sure Ollama is running on `localhost:11434`.

---

## Installation

```bash
git clone <repository-url>
cd launch_model_prediction
uv sync
uv pip install -e .
```

Create the required data directories:

```bash
mkdir -p data/images data/json models
```

> **Privacy Note:** The `data/` and `models/` directories contain sensitive information (scanned images, personal names, attendance data, ML artifacts). They are **tracked as empty folders only** — all contents inside are gitignored. The `.gitkeep` files preserve the folder structure for new contributors.

## Quick Start

### 1. Parse Menu Images

Place scanned menu images in `data/images/` and parse them into JSON:

```bash
# Single image
uv run python -m app parse data/images/menu_2026-04-14.jpg

# Batch parse all images in a directory
uv run python -m app parse-batch --images-dir data/images --output-dir data/json
```

Parsed JSON files are saved to `data/json/`.

---

### 2. Train the Model

Requires **at least 20 weeks** of historical data:

```bash
uv run python -m app train --cross-validate
```

- Runs TimeSeriesSplit Cross-Validation before final training
- Saves the trained model to `models/predictor.jobml`

---

### 3. Predict Attendance

```bash
uv run python -m app predict 2026-05
```

Output: a Rich table with predicted attendance per weekday category.

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `app parse <image>` | Parse a single scanned menu image into structured JSON. |
| `app parse-batch` | Parse all images in a directory into JSON. |
| `app train` | Train the ML model on historical JSON data. |
| `app predict <YYYY-MM>` | Predict attendance for a given month (e.g. `2026-05`). |
| `app evaluate` | Evaluate the saved model on held-out test data. |
| `app status` | Show system status: data weeks available, model readiness. |

### Example Usage

```bash
# Check system readiness
uv run python -m app status

# Train with cross-validation
uv run python -m app train --cross-validate

# Predict May 2026
uv run python -m app predict 2026-05

# Evaluate model performance
uv run python -m app evaluate --test-size 0.2
```

---

## Synthetic Data (For Testing)

To generate fake historical data (25 weeks) for development/testing:

```bash
uv run python scripts/generate_synthetic.py
```

This creates JSON files in `data/json/` with realistic weekday attendance patterns.

---

## Project Architecture

The project follows **Feature-Sliced Design (FSD)** with a strict dependency hierarchy:

```
app (CLI + Config + DI Container)
  ├── features/
  │   ├── ocr_parser/      # Image → Raw OCR Text → MenuSheet JSON
  │   ├── data_ingestion/  # JSON → FeatureEngineered DataFrame
  │   └── prediction/      # Train / Evaluate / Predict
  ├── entities/
  │   └── menu_sheet.py    # Pydantic domain models
  └── shared/
      ├── constants.py     # Paths, labels, Ollama config
      ├── exceptions.py    # Custom exceptions
      ├── logger.py        # Structured logging
      └── utils.py         # ANSI cleanup, helpers
```

**Data Flow:**

1. `data/images/*.jpg` → OCR (`deepseek-ocr:3b` / `qwen3-vl`) → Raw text
2. `RawTextParser` → `MenuSheet` (Pydantic) → `data/json/*.json`
3. `MenuRepository` + `FeatureEngineer` → `pd.DataFrame` with temporal/holiday features
4. `ModelTrainer` (`MultiOutputRegressor` + `RandomForestRegressor`) → `models/predictor.jobml`
5. `predict` command loads model → predicts per weekday → outputs Rich table

---

## Configuration

Edit `shared/constants.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_PRIMARY_MODEL` | `deepseek-ocr:3b` | Primary OCR model |
| `OLLAMA_FALLBACK_MODEL` | `qwen3-vl:235b-cloud` | Fallback OCR model |
| `DATA_DIR` | `data/` | Root data directory |
| `IMAGES_DIR` | `data/images/` | Input images |
| `JSON_DIR` | `data/json/` | Parsed JSON output |
| `MODELS_DIR` | `models/` | Serialized ML models |
| `CATEGORY_LABELS` | `["Erw", "Ki", "MA", "MA-Ki"]` | Attendance categories |

> **Note:** If `deepseek-ocr:3b` returns only layout tags (`<|ref|>`, `<|det|>`) without text, the system automatically falls back to `qwen3-vl`.

---

## OCR Notes

- The primary model `deepseek-ocr:3b` uses the prompt:  
  `Extract the text in the image.\n<|grounding|>Given the layout of the image.`
- OCR output is cleaned of ANSI escape codes, HTML tags (`<|ref|>`, `<|det|>`, `<|vq|>`), and TTY noise before parsing.
- If structured JSON generation fails, `RawTextParser` uses regex heuristics to extract names, numbers, and totals from the raw OCR text.

---

## Data Requirements

- **Minimum 20 weeks** of historical menu sheets for reliable model training.
- Each sheet should contain:
  - Weekday names (`Montag`, `Dienstag`, `Mittwoch`, `Donnerstag`, `Freitag`)
  - Summary totals for `Erw`, `Ki`, `MA`, `MA-Ki`
  - Closed days marked as `geschlossen`

---

## Development

### Run Tests

```bash
uv run pytest tests/
```

### Linting

```bash
uv run ruff check .
uv run ruff format .
```

### Project Metadata

- **Package Manager:** `uv` (see `pyproject.toml`)
- **Line Length:** 100 characters (`ruff`)
- **Python Target:** 3.12+

---

## Tech Stack

| Component | Tool |
|-----------|------|
| CLI Framework | Typer + Rich |
| Data Validation | Pydantic 2.x |
| OCR / AI | Ollama (`deepseek-ocr:3b`, `qwen3-vl`) |
| Data Processing | Pandas, NumPy |
| ML Model | scikit-learn (`MultiOutputRegressor` + `RandomForestRegressor`) |
| Serialization | joblib |
| Holidays | `holidays` (Switzerland, Bern) |
| Testing | pytest |

---

## License

This project is licensed under the **Personal Use License**.  
You may use, modify, and share the software for **personal, educational, or non-profit purposes only**.  
Commercial use, redistribution for profit, or incorporation into commercial products is **strictly prohibited** without explicit written permission.

See [LICENSE](LICENSE) for full terms.
