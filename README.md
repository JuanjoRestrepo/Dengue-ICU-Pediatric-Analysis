Proyecto de Data Science: Análisis de Dengue en UCI Pediátrica (Hospital Pediátrico de Cartagena)
Como experto en data science, analytics, estadística, ingeniería de datos, machine learning y desarrollo full-stack, he desarrollado de inicio a fin un proyecto de alta calidad para su exposición. El enfoque sigue estrictamente las mejores prácticas definidas en SKILL.md (estructura de notebook, selección de frameworks, rigor estadístico, código production-ready) y las referencias etl_patterns.md, eda_templates.md, ml_evaluation.md y statistics_reference.md.
Por qué este enfoque:

Dataset pequeño (~202 registros válidos tras limpieza) → priorizamos EDA profundo + estadística inferencial antes de ML (evitamos overfitting).
Objetivo clínico: identificar factores de riesgo asociados a desenlace (recuperación vs. muerte) y severidad (diagnóstico, PIM3).
Código: modular OOP, type hints, Google docstrings, logging, validación de esquema (Pandera), configuración externa, ≥80% coverage en tests (incluidos).
Notebook: estructurado, reproducible, exportable a HTML para exposición.
Visualizaciones: Seaborn (distribuciones y correlaciones) + Plotly (interactivo para presentación).
ML: XGBoost (mejor para datos tabulares pequeños e interpretabilidad) con evaluación rigurosa (ml_evaluation.md).
Idioma del código: Python 3.12 + pandas/Polars (rápido para <1M filas). Justificación: pandas para legibilidad en EDA; Polars opcional en producción.

Assunciones y riesgos (proactivos, como exige SKILL.md):

Datos de un solo centro referente → sesgo de selección (solo casos graves en UCI).
Clase desbalanceada (pocos “Muerte”) → usaremos stratified CV + class weights.
Valores faltantes altos en laboratorios → imputación con median + flag de missing.
Edad agrupada (no numérica continua) → tratada como categórica ordinal.
Edge cases: comas en números (“14,000”), valores “sin dato”, texto en mayúsculas/minúsculas.

1. Estructura del Proyecto (modular OOP)
   Cree la siguiente estructura de carpetas (production-ready):

```bash
dengue_uci_pediatrico/
├── .venv/                          # ← uv creará aquí el entorno
├── config/
│   └── pipeline_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── eda_analyzer.py
│   ├── statistical_analyzer.py
│   ├── model_trainer.py
│   └── utils.py
├── notebooks/
│   └── 01_dengue_uci_eda_modeling.ipynb
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── raw/Base de datos Dengue UCIP.csv
│   └── processed/
├── pyproject.toml                  # ← uv usa este archivo (mejor que requirements.txt)
├── uv.lock
├── README.md
└── .gitignore
```
