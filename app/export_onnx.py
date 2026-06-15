"""
best.pt → best.onnx 변환 (최초 1회만 실행)
사용법:
    pip install ultralytics
    python export_onnx.py --weights models/best.pt
"""
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best.pt", help="학습된 .pt 가중치 경로")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    # models/best.onnx 위치로 정리
    out = Path(out)
    target = Path("models") / "best.onnx"
    target.parent.mkdir(exist_ok=True)
    if out.resolve() != target.resolve():
        target.write_bytes(out.read_bytes())
    print(f"[done] ONNX 저장 → {target}")
    print("이제 python app/main.py 로 프로그램을 실행할 수 있습니다.")

if __name__ == "__main__":
    main()
