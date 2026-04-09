# Dengue-ICU-Pediatric-Analysis

## Project Overview & Objectives

Proyecto de Data Science de inicio a fin sobre el análisis de Dengue en la UCI Pediátrica del Hospital Pediátrico de Cartagena.
Como experto en Data Science y Full-Stack, he desarrollado este proyecto bajo estándares de alta calidad, siguiendo rigor estadístico y código _production-ready_.

- **Main Objective:** Identify clinical factors associated with severity and outcomes (recovery vs. death).
- **Key Constraints:** Small dataset (~200 samples) with high-dimensional clinical variables.
- **Core Approach:** Priorizamos un **EDA profundo y estadística inferencial** (Mann-Whitney U, Chi², Spearman) antes que el modelado masivo para mitigar el riesgo de _overfitting_.

## Why This Approach?

- **Dataset Size:** (~202 registros válidos) → El rigor estadístico es nuestra brújula para evitar falsos descubrimientos.
- **ML Tooling:** XGBoost (seleccionado por su manejo de datos tabulares y facilidad de interpretabilidad) empleado como herramienta complementaria.
- **Code Standards:** Modular OOP, Type Hints (Python 3.12), Google/NumPy docstrings, logging y validación de esquemas con Pandera.
- **Visualizations:** Mix de Seaborn para rigor analítico y Plotly para interactividad en la exposición.

## Project Structure (Modular OOP)

```bash
Dengue-ICU-Pediatric-Analysis/
├── config/
│   └── pipeline_config.yaml
├── data/
│   ├── processed/
│   │   └── dengue_uci_cleaned.csv
│   └── raw/
│       └── Base_de_Datos_Dengue_UCIP.csv
├── notebooks/
│   └── 01_dengue_uci_eda_modeling.ipynb
├── src/
│   ├── data_cleaner.py
│   ├── data_loader.py
│   ├── eda_analyzer.py
│   ├── model_trainer.py
│   ├── statistical_analyzer.py
│   ├── utils.py
│   └── __init__.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## Assumptions & Risks (Proactive Analysis)

- **Selection Bias:** Casos provenientes de un solo centro de referencia; representa únicamente la población de cuidados intensivos.
- **Class Imbalance:** Fuerte desbalance en la variable objetivo (pocos eventos de "Muerte"); se utiliza stratified CV y class weights.
- **Data Quality:** Valores faltantes imputados mediante mediana con flagging de registros nulos para mantener la integridad del análisis.

## Development Stack

- **Environment**: uv (fast, reliable dependency management).
- **Linter/Formatter**: Ruff (PEP8 compliance).
- **Type Checker**: Mypy & Pylance (Strict typing).
- **CI/CD Ready**: Modular architecture prepared for automated testing.
