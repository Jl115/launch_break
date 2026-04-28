# Lunch Attendance Prediction System — Architecture

## Directory Tree

```
launch_model_prediction/
├── AGENTS.md          # Coding conventions
├── ARCHITECTURE.md    # This file
├── pyproject.toml     # uv project config
├── data/
│   ├── images/        # Raw scanned menu sheets (.jpg)
│   └── json/          # Parsed structured output (.json)
├── models/            # Serialized ML artifacts (.jobml)
├── app/
│   ├── __main__.py    # Entry point
│   ├── cli.py         # Typer CLI commands
│   ├── config.py      # Pydantic Settings
│   └── container.py   # Manual DI container
├── entities/
│   ├── menu_sheet.py  # Pydantic: MenuSheet, DailyMenu, Signup
│   └── prediction_result.py
├── features/
│   ├── ocr_parser/    # Image → Raw Text → JSON
│   ├── data_ingestion/# JSON → DataFrame → ML features
│   └── prediction/    # Train / Evaluate / Predict
├── shared/
│   ├── logger.py
│   ├── exceptions.py
│   ├── constants.py
│   └── utils.py
└── tests/
    ├── unit/
    └── integration/
```

## Dependency Flow

```
app (CLI + Config + DI)
  ├─▶ features/
  │      ├─ ocr_parser     (entities, shared)
  │      ├─ data_ingestion (entities, shared)
  │      └─ prediction     (entities, shared)
  │
  ├─▶ entities (menu_sheet, prediction_result)
  │
  └─▶ shared (logger, exceptions, constants, utils)
```

**No reverse arrows permitted.**

## Data Flow

1. **OCR** (`ocr_parser`):
   - `data/images/*.jpg` → `OllamaClient` (deepseek-ocr:3b) → Raw text
   - `RawTextParser` / Regex → `MenuSheet` (Pydantic)
   - Saved to `data/json/YYYY-MM-DD.json`

2. **Ingestion** (`data_ingestion`):
   - `data/json/*.json` → `MenuRepository`
   - `FeatureEngineer` builds temporal + holiday + weekday features
   - Output: `pd.DataFrame` for ML

3. **Prediction** (`prediction`):
   - Input: `pd.DataFrame`
   - `ModelTrainer`: `MultiOutputRegressor` + `RandomForestRegressor`
   - `ModelEvaluator`: MAE / RMSE per category
   - Trained artifact saved to `models/predictor.jobml`

4. **Prediction Command** (`app/cli.py`):
   - Loads model + holiday calendar
   - Predicts for each weekday of target month
   - Prints Rich table (Erw, Ki, MA, MA-Ki)
