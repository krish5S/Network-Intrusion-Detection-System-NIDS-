#!/usr/bin/env bash
# Quick launcher for NIDS Sentinel Streamlit Frontend
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting NIDS Sentinel Cybersecurity SOC Console..."
streamlit run app.py --server.port 8501 --server.address localhost
