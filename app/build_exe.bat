@echo off
REM === PCB Defect Inspector exe 빌드 (윈도우) ===
REM 사전: pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name PCB-Inspector ^
  --add-data "models\best.onnx;models" ^
  app\main.py
echo.
echo [done] dist\PCB-Inspector\PCB-Inspector.exe 생성됨
pause
