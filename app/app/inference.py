"""PCB 결함 검출 추론 엔진 (ONNX Runtime, CPU)
- YOLO26 end2end 출력(1,300,6)과 일반 YOLO raw 출력(1, 4+nc, N) 모두 지원
- 이중 임계값 판정: conf >= t_high → 자동 불량(AUTO_NG)
                    t_low <= conf < t_high → 사람 재검(REVIEW)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ---- 추론 하이퍼파라미터 (모델 학습/익스포트 설정과 일치) ----
DEFAULT_IMGSZ = 640          # 모델 입력 정사각 크기 (YOLO 학습 imgsz)
LETTERBOX_PAD = 114          # letterbox 패딩 색 (YOLO 표준 회색)
DEFAULT_IOU = 0.5            # raw 출력 후처리 시 NMS IoU 임계값
DEFAULT_CONF_MIN = 0.05      # 후처리에 남길 최소 confidence

CLASS_NAMES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
CLASS_KOR = {
    "SH": "단락", "SP": "스퍼", "SC": "잔여구리", "OP": "단선", "MB": "마우스바이트",
    "HB": "홀 브레이크아웃", "CS": "스크래치", "CFO": "도체 이물", "BMFO": "기판 이물",
}


@dataclass
class Detection:
    """검출 1건. 좌표는 원본 이미지 픽셀 기준(letterbox 역변환 후)."""
    x1: float; y1: float; x2: float; y2: float
    conf: float
    cls_id: int

    @property
    def name(self) -> str:
        return CLASS_NAMES[self.cls_id] if 0 <= self.cls_id < len(CLASS_NAMES) else str(self.cls_id)


def letterbox(img: np.ndarray, size: int = DEFAULT_IMGSZ):
    """비율 유지 리사이즈 + 패딩으로 정사각(size x size) 입력을 만든다.

    이미지를 종횡비를 유지한 채 size 안에 맞춰 축소/확대하고, 남는 영역을
    회색(LETTERBOX_PAD)으로 패딩해 중앙 정렬한다. 역변환에 필요한 스케일(r)과
    좌/상단 패딩 오프셋(left, top)을 함께 반환한다.

    Args:
        img: HxWx3 이미지 (BGR/RGB 무관).
        size: 출력 정사각 한 변 길이.

    Returns:
        (canvas, r, left, top) — 패딩된 이미지, 적용 스케일, x/y 패딩 픽셀.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = DEFAULT_IOU):
    """탐욕적(greedy) Non-Maximum Suppression.

    점수가 높은 박스부터 순서대로 채택하고, 채택된 박스와 IoU가 iou_thr 이상
    겹치는 나머지 박스를 제거한다. 후보가 없을 때까지 반복하여 중복 검출을
    하나로 합친다.

    Args:
        boxes: (N,4) [x1,y1,x2,y2].
        scores: (N,) 각 박스의 confidence.
        iou_thr: 이 IoU 이상 겹치면 억제.

    Returns:
        채택된 박스의 인덱스 리스트(점수 내림차순).
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (a1 + a2 - inter + 1e-9)
        idxs = idxs[1:][iou < iou_thr]
    return keep


class PCBDetector:
    """ONNX Runtime(CPU) 기반 PCB 결함 검출기.

    best.onnx 모델을 로드해 단일 이미지를 추론한다. YOLO26 end2end 출력과
    일반 raw 출력을 모두 처리하며, raw 출력에는 자체 NMS를 적용한다.
    """

    def __init__(self, onnx_path: str | Path, imgsz: int = DEFAULT_IMGSZ):
        try:
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception as e:  # 파일 없음/손상/형식 오류 등
            raise RuntimeError(f"ONNX model load failed: {onnx_path} ({e})") from e
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz

    def predict(self, img_bgr: np.ndarray, conf_min: float = DEFAULT_CONF_MIN) -> list[Detection]:
        """단일 BGR 이미지를 추론해 검출 리스트(confidence 내림차순)를 반환한다.

        letterbox로 전처리 → ONNX 추론 → 출력 형식(end2end / raw)에 맞춰 후처리
        → 좌표를 원본 이미지 픽셀로 역변환하고 이미지 경계로 클리핑한다.

        Args:
            img_bgr: OpenCV BGR 이미지 (HxWx3).
            conf_min: 이 confidence 미만의 검출은 버린다.

        Returns:
            Detection 리스트 (conf 내림차순).
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp, r, padx, pady = letterbox(rgb, self.imgsz)
        blob = inp.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.input_name: blob})[0]

        dets: list[Detection] = []
        if out.ndim == 3 and out.shape[-1] == 6:
            # end2end: (1, N, 6) = x1,y1,x2,y2,conf,cls
            for x1, y1, x2, y2, cf, c in out[0]:
                if cf < conf_min:
                    continue
                dets.append(Detection((x1 - padx) / r, (y1 - pady) / r,
                                      (x2 - padx) / r, (y2 - pady) / r, float(cf), int(c)))
        else:
            # raw: (1, 4+nc, N)
            pred = out[0]
            if pred.shape[0] < pred.shape[1]:
                pred = pred  # (4+nc, N)
            else:
                pred = pred.T
            boxes_xywh = pred[:4].T          # (N,4) cx,cy,w,h
            cls_scores = pred[4:].T          # (N,nc)
            confs = cls_scores.max(1)
            clss = cls_scores.argmax(1)
            m = confs >= conf_min
            boxes_xywh, confs, clss = boxes_xywh[m], confs[m], clss[m]
            if len(boxes_xywh):
                xy = boxes_xywh[:, :2]; wh = boxes_xywh[:, 2:]
                boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
                keep = _nms(boxes, confs, DEFAULT_IOU)
                for i in keep:
                    x1, y1, x2, y2 = boxes[i]
                    dets.append(Detection((x1 - padx) / r, (y1 - pady) / r,
                                          (x2 - padx) / r, (y2 - pady) / r,
                                          float(confs[i]), int(clss[i])))
        h, w = img_bgr.shape[:2]
        for d in dets:
            d.x1 = max(0.0, min(d.x1, w - 1)); d.x2 = max(0.0, min(d.x2, w - 1))
            d.y1 = max(0.0, min(d.y1, h - 1)); d.y2 = max(0.0, min(d.y2, h - 1))
        dets.sort(key=lambda d: -d.conf)
        return dets


def verdict(dets: list[Detection], t_low: float, t_high: float) -> str:
    """이미지 단위 판정: AUTO_NG / REVIEW / PASS"""
    if any(d.conf >= t_high for d in dets):
        return "AUTO_NG"
    if any(t_low <= d.conf < t_high for d in dets):
        return "REVIEW"
    return "PASS"
