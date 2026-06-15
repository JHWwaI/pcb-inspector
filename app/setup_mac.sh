#!/bin/bash
# === 맥북 셋업 (최초 1회) — 터미널에서: bash setup_mac.sh ===
set -e
cd "$(dirname "$0")"

echo "[1/3] 가상환경 + 패키지 설치"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt ultralytics

echo "[2/3] best.pt → best.onnx 변환"
python export_onnx.py --weights models/best.pt

echo "[3/3] 완료! 실행:"
echo "    bash run_mac.sh"
