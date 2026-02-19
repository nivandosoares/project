#!/usr/bin/env bash
set -euo pipefail

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
