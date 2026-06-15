# PCB Defect Inspector — app/

실행 가능한 애플리케이션 코드와 학습 모델이 모두 이 폴더에 있다.
프로젝트 전체 개요·결과·구조는 루트 [`../README.md`](../README.md)를 참고할 것.

## 빠른 실행

```bash
# 런타임 의존성 설치
pip install -r requirements.txt      # 또는: uv pip install -r requirements.txt

# GUI 실행 (models/best.onnx 필요)
python app/main.py
```

- 파일 메뉴 → 이미지/폴더 열기 → 자동 검사
- 좌측: 파일별 판정 리스트(색상) / 중앙: 박스 오버레이(휠 줌·드래그)
- 우측: t_low/t_high 슬라이더(실시간 반영), 결함 테이블, CSV 내보내기

## best.onnx 생성 (최초 1회)

`models/best.onnx`가 없을 때만 `best.pt`에서 변환한다.

```bash
pip install -r requirements-dev.txt
python export_onnx.py --weights models/best.pt
```

## 배포 빌드

```bash
build_exe.bat            # Windows → dist/PCB-Inspector/PCB-Inspector.exe
bash build_app_mac.sh    # macOS   → dist/PCB-Inspector/
```

받은 사람은 압축을 풀고 실행 파일을 더블클릭하면 된다 (파이썬 설치 불필요).

## macOS 원클릭 스크립트

```bash
bash setup_mac.sh        # 최초 1회: venv + 패키지 + ONNX 변환
bash run_mac.sh          # 실행
bash build_app_mac.sh    # (선택) .app 빌드
```

## 판정 로직 (이중 임계값)

| 구간 | 판정 |
|---|---|
| conf ≥ t_high (기본 0.70) | 자동 불량 — 사람 확인 생략 |
| t_low ≤ conf < t_high | 재검 필요 — 사람이 확인 |
| conf < t_low (기본 0.25) | 보고 안 함 |
