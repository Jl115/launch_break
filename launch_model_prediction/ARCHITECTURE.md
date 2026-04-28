# Architecture

## Directory Tree

```
launch_model_prediction/
├── app/                                # Application layer (thin CLI + TUI)
│   ├── __init__.py
│   ├── __main__.py                     # Entry point dispatches TUI or CLI
│   ├── actions.py                      # Reusable business actions (CLI & TUI)
│   ├── cli.py                          # Typer CLI commands
│   ├── config.py                       # Pydantic settings
│   ├── container.py                    # Manual DI container
│   └── tui_app.py                      # Textual TUI application (plotext charts)
├── entities/                            # Domain models
│   ├── __init__.py
│   └── menu_sheet.py                   # Pydantic models (MenuSheet, DailyMenu, etc.)
├── features/                            # Feature-Sliced Design modules
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── repository.py               # JSON loading & repository pattern
│   │   └── engineer.py                 # Feature engineering (DataFrame creation)
│   ├── ocr_parser/
│   │   ├── __init__.py
│   │   ├── ollama_client.py            # Ollama HTTP client for OCR
│   │   ├── raw_parser.py               # Heuristic text → MenuSheet parser
│   │   └── service.py                  # Orchestrates OCR + parsing
│   └── prediction/
│       ├── __init__.py
│       ├── evaluator.py                # Model evaluation metrics (MAE, RMSE)
│       ├── model.py                    # SklearnLunchPredictor (RandomForest)
│       └── trainer.py                  # ModelTrainer + cross-validation
├── shared/                              # Shared utilities (no upstream deps)
│   ├── __init__.py
│   ├── constants.py                    # Paths, config constants
│   ├── exceptions.py                   # Domain exceptions
│   ├── logger.py                       # get_logger factory
│   └── utils.py                        # OCR cleaning helpers
├── scripts/
│   └── generate_synthetic.py           # Generate fake menu sheets for testing
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   └── __init__.py
│   └── unit/
│       └── __init__.py
├── data/
│   ├── images/                         # Input scanned menu sheets
│   └── json/                           # Parsed JSON output
├── models/                              # Serialized ML artifacts
├── pyproject.toml                       # uv / hatch build config
└── ARCHITECTURE.md                      # This file
```

## Architectural Decisions

- **Feature-Sliced Design (FSD)**: Strict dependency direction `app -> features -> entities -> shared`
- **DI Container**: Manual wiring in `app/container.py`
- **CLI/TUI Duality**: Business logic extracted to `app/actions.py`; `app/cli.py` (Typer) and `app/tui_app.py` (Textual) are thin consumers.
- **Entry Point**: `python -m app` → Textual TUI when no arguments; forwards to Typer CLI when arguments are present.

## Data Flow

1. **Images** → `features/ocr_parser/ollama_client.py` → Raw text
2. Raw text → `features/ocr_parser/raw_parser.py` → `MenuSheet` entity
3. `MenuSheet` → `features/data_ingestion/repository.py` → JSON on disk
4. JSON → `features/data_ingestion/engineer.py` → `pd.DataFrame`
5. DataFrame → `features/prediction/trainer.py` → Trained model artifact
6. Model artifact → `features/prediction/model.py` → Predictions

## Verification

Run syntax check:
```bash
uv run python -m compileall app/ entities/ features/ shared/ scripts/ -q
```

Run CLI commands:
```bash
uv run python -m app status
uv run python -m app data
uv run python -m app predict 2026-04
```

Launch TUI:
```bash
uv run python -m app
```
