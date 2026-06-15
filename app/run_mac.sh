#!/bin/bash
# === 프로그램 실행 — 터미널에서: bash run_mac.sh ===
cd "$(dirname "$0")"
source .venv/bin/activate
python app/main.py
