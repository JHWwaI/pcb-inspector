#!/bin/bash
# === (선택) 맥용 배포 앱(.app) 빌드 ===
cd "$(dirname "$0")"
source .venv/bin/activate
pip install -q pyinstaller
pyinstaller --noconfirm --windowed --name PCB-Inspector \
  --add-data "models/best.onnx:models" \
  app/main.py
echo "[done] dist/PCB-Inspector/ 생성 — PCB-Inspector 실행 파일 더블클릭"
