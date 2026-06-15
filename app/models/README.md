# 모델 파일

| 파일 | 형식 | 용도 |
|---|---|---|
| `best.pt` | PyTorch (YOLO26s) | DsPCBSD+ 파인튜닝 학습 가중치. mAP@50 **0.843** (9 클래스). 재검증·재학습·ONNX 익스포트의 원본. |
| `best.onnx` | ONNX | `best.pt`를 `export_onnx.py`로 변환한 CPU 추론용 모델. 앱 런타임이 ONNX Runtime(CPU)으로 로드한다. |

학습: Ultralytics YOLO26s를 DsPCBSD+ PCB 표면 결함 데이터셋(9 클래스)으로 파인튜닝, imgsz 640.

`best.onnx`는 다음으로 재생성할 수 있다:

```bash
python export_onnx.py --weights models/best.pt
```
