# AI Pyramid Risk Detection MVP

## Overview
APRIS is a local AI prototype for early detection of financial pyramid schemes.
The system uses a synthetic population of organizations, computes risk scores with a trained RandomForest model, and provides analyst-facing visual diagnostics in Streamlit.
It includes single-case scoring, dynamic signal breakdown, synthetic transaction graph analysis, and a PCA-based population risk map.
All inference is executed locally without external AI APIs.
The MVP is designed for hackathon demonstration and fast extension to real compliance workflows.

## Problem
Financial pyramids often imitate legitimate growth in early stages, which makes manual detection slow and inconsistent. Monitoring teams need a repeatable way to prioritize suspicious entities before collapse events. APRIS addresses this by combining behavioral and structural indicators into an interpretable risk score with visual evidence for analysts.

## Architecture
- `data_generator.py` - synthetic dataset generation with borderline cases and correlated features.
- `train_model.py` - model training and artifact export.
- `risk_engine.py` - artifact loading, input validation, risk scoring, and feature explanation.
- `graph_module.py` - synthetic transaction graph generation and hub metrics.
- `population_map.py` - PCA projection of synthetic population with current-case overlay.
- `app.py` (Streamlit UI) - analyst dashboard and demo workflow.

## Model
- Algorithm: `RandomForestClassifier`
- Test recall (pyramid, label=1): `0.96`
- Test ROC-AUC: `0.993`
- Data design: synthetic population with `15%` borderline cases and overlapping class distributions.

## How to Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- Synthetic data only (no real banking data in this MVP).
- No direct integration with banking core systems or regulator data feeds.

## Roadmap
- Graph Neural Networks (GNN) for transaction-topology risk learning.
- NLP pipeline for complaint/news/investigation text signals.
- CBDC and digital-asset transaction compatibility layer.
