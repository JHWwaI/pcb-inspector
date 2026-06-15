"""best.pt 로 DsPCBSD+ val 세트 mAP 검증 — 0.843 재현 + confusion matrix/PR curve 생성."""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "models" / "best.pt"
DATA = HERE.parent / "_dataset_tmp" / "yolo_val" / "data.yaml"
DOCS = HERE.parent / "docs" / "metrics"
DOCS.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))
m = model.val(data=str(DATA), imgsz=640, batch=4, conf=0.001, iou=0.6,
              split="val", plots=True, save_json=False, workers=0, device="cpu",
              project=str(HERE / "_val_run"), name="val", exist_ok=True)

print("\n========== 검증 결과 ==========")
print(f"mAP@50    : {m.box.map50:.4f}")
print(f"mAP@50-95 : {m.box.map:.4f}")
print(f"precision : {m.box.mp:.4f}")
print(f"recall    : {m.box.mr:.4f}")
print("클래스별 mAP@50:")
names = m.names
for i, ap in enumerate(m.box.ap50):
    print(f"  {names[i]:5s} {ap:.4f}")

# 산출 차트를 docs/metrics 로 복사
run = HERE / "_val_run" / "val"
for f in ["confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png",
          "P_curve.png", "R_curve.png", "F1_curve.png", "val_batch0_pred.jpg"]:
    src = run / f
    if src.exists():
        shutil.copy(src, DOCS / f)
print(f"\n차트 -> {DOCS}")
