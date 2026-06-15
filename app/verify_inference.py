"""헤드리스 검증 — best.onnx 로 PCB 결함을 검출하고 오버레이·판정·CSV 생성.
사용법: python verify_inference.py <이미지_또는_폴더> [출력폴더]
GUI 없이 추론 엔진(app/inference.py)의 실동작을 증명한다."""
import sys, csv, time
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "app"))
from inference import PCBDetector, verdict, CLASS_KOR, CLASS_NAMES  # noqa: E402

MODEL = HERE / "models" / "best.onnx"
T_LOW, T_HIGH = 0.25, 0.70

# 클래스별 색 (BGR)
rng = np.random.default_rng(7)
COLORS = {i: tuple(int(x) for x in rng.integers(60, 235, 3)) for i in range(len(CLASS_NAMES))}
VERDICT_COLOR = {"AUTO_NG": (40, 40, 235), "REVIEW": (40, 175, 240), "PASS": (90, 190, 90)}


def draw(img, dets, vd):
    out = img.copy()
    for d in dets:
        if d.conf < T_LOW:
            continue
        c = COLORS[d.cls_id]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, c, 2)
        label = f"{d.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(out, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # 상단 판정 배지
    vc = VERDICT_COLOR[vd]
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), vc, -1)
    cv2.putText(out, f"{vd}  |  defects: {sum(1 for d in dets if d.conf >= T_LOW)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_dataset_tmp"
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "docs" / "screenshots" / "detections"
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = [src] if src.is_file() else sorted(
        [p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    )
    det = PCBDetector(MODEL)
    print(f"model: {MODEL.name} | images: {len(imgs)}")

    rows = []
    t0 = time.time()
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        t = time.time()
        dets = det.predict(img, conf_min=0.05)
        ms = (time.time() - t) * 1000
        vd = verdict(dets, T_LOW, T_HIGH)
        shown = [d for d in dets if d.conf >= T_LOW]
        cv2.imwrite(str(outdir / f"det_{i:02d}_{vd}.jpg"), draw(img, dets, vd))
        cls_summary = ",".join(f"{d.name}:{d.conf:.2f}" for d in shown[:6])
        rows.append([p.name, vd, len(shown), f"{ms:.0f}", cls_summary])
        print(f"[{i:02d}] {p.name:22.22s} -> {vd:8s} ({len(shown)} defects, {ms:.0f}ms) {cls_summary}")

    with open(outdir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "verdict", "defects", "ms", "classes"])
        w.writerows(rows)

    avg = (time.time() - t0) / max(len(rows), 1) * 1000
    ng = sum(1 for r in rows if r[1] == "AUTO_NG")
    rv = sum(1 for r in rows if r[1] == "REVIEW")
    ps = sum(1 for r in rows if r[1] == "PASS")
    print(f"\n=== {len(rows)}장 | AUTO_NG {ng} · REVIEW {rv} · PASS {ps} | 평균 {avg:.0f}ms/장 ===")
    print(f"오버레이·CSV -> {outdir}")


if __name__ == "__main__":
    main()
