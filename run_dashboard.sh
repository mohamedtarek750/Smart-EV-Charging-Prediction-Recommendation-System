#!/usr/bin/env bash
# Launches the dashboard through the same interpreter that trained the models.
cd "$(dirname "$0")"
python -m streamlit run app/dashboard.py
