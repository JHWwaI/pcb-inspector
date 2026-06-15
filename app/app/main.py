"""PCB Defect Inspector — PySide6 데스크톱 검사 프로그램.

PCB 이미지를 열어 ONNX 추론 엔진(inference.PCBDetector)으로 결함을 검출하고,
이중 임계값(t_low/t_high)으로 자동 불량/재검/통과를 판정해 시각화한다.

실행:  python app/main.py          (models/best.onnx 필요)
"""
from __future__ import annotations
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QColor, QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QPushButton,
)

# main.py는 app/app/main.py 에 위치 → APP_DIR = .../app, PKG_DIR = .../app/app.
# 패키지로 실행되지 않고 스크립트로 직접 실행될 때를 대비해 inference 모듈을
# import 할 수 있도록 패키지 디렉터리를 sys.path 에 추가한다.
PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent
sys.path.insert(0, str(PKG_DIR))
from inference import PCBDetector, verdict, CLASS_KOR  # noqa: E402

MODEL_PATH = APP_DIR / "models" / "best.onnx"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ---- UI / 판정 상수 ----
WINDOW_W, WINDOW_H = 1280, 800       # 메인 창 기본 크기
DEFAULT_T_LOW = 0.25                 # 재검 하한 임계값 (이 미만은 보고 안 함)
DEFAULT_T_HIGH = 0.70                # 자동 불량 임계값 (이 이상은 NG 확정)
SLIDER_MIN, SLIDER_MAX = 5, 95       # 임계값 슬라이더 범위 (백분율)

VERDICT_STYLE = {
    "AUTO_NG": ("자동 불량", "#D85A30"),
    "REVIEW":  ("재검 필요", "#E8A33D"),
    "PASS":    ("통과",     "#1D9E75"),
}


class ImageView(QGraphicsView):
    """줌/패닝 가능한 이미지 뷰어"""
    def __init__(self):
        super().__init__()
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def show_image(self, pix: QPixmap):
        """픽스맵을 씬에 표시하고 뷰에 맞춰 비율 유지로 맞춘다."""
        self.scene_.clear()
        self.scene_.addItem(QGraphicsPixmapItem(pix))
        self.setSceneRect(QRectF(pix.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    """메인 창 — 파일 리스트 · 이미지 뷰어 · 임계값/결함 테이블을 구성하고
    이미지 배치 검사와 판정 시각화를 담당한다."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Defect Inspector — YOLO26s")
        self.resize(WINDOW_W, WINDOW_H)

        self.detector: PCBDetector | None = None
        self.results: dict[str, list] = {}     # path -> detections
        self.t_low, self.t_high = DEFAULT_T_LOW, DEFAULT_T_HIGH

        # ---- 좌: 파일 리스트 ----
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_select)

        # ---- 중앙: 이미지 뷰어 ----
        self.viewer = ImageView()
        self.verdict_label = QLabel("이미지를 열어주세요")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet("font-size:18px; font-weight:bold; padding:6px;")

        # ---- 우: 임계값 + 결함 테이블 ----
        self.low_slider = self._slider(int(self.t_low * 100))
        self.high_slider = self._slider(int(self.t_high * 100))
        self.low_label = QLabel(); self.high_label = QLabel()
        self.low_slider.valueChanged.connect(self.on_threshold)
        self.high_slider.valueChanged.connect(self.on_threshold)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["클래스", "한글명", "conf", "판정"])
        self.table.horizontalHeader().setStretchLastSection(True)

        export_btn = QPushButton("결과 CSV 내보내기")
        export_btn.clicked.connect(self.export_csv)

        right = QWidget(); rl = QVBoxLayout(right)
        rl.addWidget(QLabel("재검 하한 임계값 (t_low)")); rl.addWidget(self.low_slider); rl.addWidget(self.low_label)
        rl.addWidget(QLabel("자동 불량 임계값 (t_high)")); rl.addWidget(self.high_slider); rl.addWidget(self.high_label)
        rl.addWidget(QLabel("검출된 결함")); rl.addWidget(self.table); rl.addWidget(export_btn)

        center = QWidget(); cl = QVBoxLayout(center)
        cl.addWidget(self.verdict_label); cl.addWidget(self.viewer)

        split = QSplitter()
        split.addWidget(self.file_list); split.addWidget(center); split.addWidget(right)
        split.setSizes([220, 760, 300])
        self.setCentralWidget(split)

        # ---- 메뉴 ----
        m = self.menuBar().addMenu("파일")
        a1 = QAction("이미지 열기...", self); a1.triggered.connect(self.open_images); m.addAction(a1)
        a2 = QAction("폴더 열기...", self); a2.triggered.connect(self.open_folder); m.addAction(a2)

        self.statusBar().showMessage("모델 로딩 중...")
        self._load_model()
        self.on_threshold()

    # ---------- helpers ----------
    def _slider(self, val):
        s = QSlider(Qt.Horizontal); s.setRange(SLIDER_MIN, SLIDER_MAX); s.setValue(val)
        return s

    def _load_model(self):
        if not MODEL_PATH.exists():
            QMessageBox.critical(self, "모델 없음",
                f"{MODEL_PATH} 가 없습니다.\n\nexport_onnx.py 로 best.pt를 변환해 models/best.onnx 를 만들어주세요.")
            self.statusBar().showMessage("모델 없음 — export_onnx.py 실행 필요")
            return
        self.detector = PCBDetector(MODEL_PATH)
        self.statusBar().showMessage("모델 로드 완료 (ONNX Runtime, CPU)")

    # ---------- file open ----------
    def open_images(self):
        """파일 선택 다이얼로그로 고른 이미지들을 배치 검사한다."""
        paths, _ = QFileDialog.getOpenFileNames(self, "이미지 선택", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if paths:
            self._run_batch([Path(p) for p in paths])

    def open_folder(self):
        """선택한 폴더 안의 모든 지원 이미지를 배치 검사한다."""
        d = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if d:
            paths = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)
            if not paths:
                QMessageBox.information(self, "알림", "폴더에 이미지가 없습니다.")
                return
            self._run_batch(paths)

    def _run_batch(self, paths: list[Path]):
        if self.detector is None:
            self._load_model()
            if self.detector is None:
                return
        self.file_list.clear(); self.results.clear()
        t0 = time.time(); n = 0
        for p in paths:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self.results[str(p)] = (img, self.detector.predict(img, conf_min=0.05))
            n += 1
            self.statusBar().showMessage(f"검사 중... {n}/{len(paths)}")
            QApplication.processEvents()
        self.statusBar().showMessage(f"완료 — {n}장 / {time.time()-t0:.1f}s")
        self._refresh_list()
        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    # ---------- threshold / list ----------
    def on_threshold(self):
        """슬라이더 값으로 t_low/t_high를 갱신하고 리스트·뷰를 다시 그린다."""
        self.t_low = self.low_slider.value() / 100
        self.t_high = max(self.high_slider.value() / 100, self.t_low + 0.01)
        self.low_label.setText(f"t_low = {self.t_low:.2f}  (이 미만은 보고 안 함)")
        self.high_label.setText(f"t_high = {self.t_high:.2f}  (이 이상은 자동 불량 확정)")
        self._refresh_list()
        self.on_select(self.file_list.currentItem())

    def _refresh_list(self):
        cur = self.file_list.currentRow()
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path, (_, dets) in self.results.items():
            v = verdict(dets, self.t_low, self.t_high)
            label, color = VERDICT_STYLE[v]
            item = QListWidgetItem(f"[{label}] {Path(path).name}")
            item.setData(Qt.UserRole, path)
            item.setForeground(QColor(color))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)
        if 0 <= cur < self.file_list.count():
            self.file_list.setCurrentRow(cur)

    # ---------- render ----------
    def on_select(self, item):
        """선택된 이미지의 박스 오버레이·판정 배지·결함 테이블을 렌더링한다."""
        if item is None:
            return
        path = item.data(Qt.UserRole)
        img, dets = self.results[path]
        vis = img.copy()
        shown = []
        for d in dets:
            if d.conf < self.t_low:
                continue
            color = (48, 90, 216) if d.conf >= self.t_high else (61, 163, 232)  # BGR
            cv2.rectangle(vis, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
            cv2.putText(vis, f"{d.name} {d.conf:.2f}", (int(d.x1), max(14, int(d.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            shown.append(d)

        v = verdict(dets, self.t_low, self.t_high)
        label, color = VERDICT_STYLE[v]
        self.verdict_label.setText(f"판정: {label}   (검출 {len(shown)}건)")
        self.verdict_label.setStyleSheet(
            f"font-size:18px; font-weight:bold; padding:6px; color:white; background:{color}; border-radius:6px;")

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.viewer.show_image(QPixmap.fromImage(qimg))

        self.table.setRowCount(len(shown))
        for r, d in enumerate(shown):
            zone = "자동 불량" if d.conf >= self.t_high else "재검"
            for c, txt in enumerate([d.name, CLASS_KOR.get(d.name, ""), f"{d.conf:.3f}", zone]):
                self.table.setItem(r, c, QTableWidgetItem(txt))

    # ---------- export ----------
    def export_csv(self):
        """현재 임계값 기준 검사 결과를 UTF-8(BOM) CSV로 저장한다."""
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "inspection_results.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "verdict", "class", "class_kor", "conf", "x1", "y1", "x2", "y2"])
            for fp, (_, dets) in self.results.items():
                v = verdict(dets, self.t_low, self.t_high)
                if not dets:
                    w.writerow([Path(fp).name, VERDICT_STYLE[v][0], "", "", "", "", "", "", ""])
                for d in dets:
                    if d.conf < self.t_low:
                        continue
                    w.writerow([Path(fp).name, VERDICT_STYLE[v][0], d.name, CLASS_KOR.get(d.name, ""),
                                f"{d.conf:.3f}", int(d.x1), int(d.y1), int(d.x2), int(d.y2)])
        self.statusBar().showMessage(f"CSV 저장 완료 → {path}")


DARK_QSS = """
* { color: #E6E6E6; font-size: 13px; }

QMainWindow, QWidget { background-color: #1F2225; }

QMenuBar { background: #2A2D31; color: #E6E6E6; border-bottom: 1px solid #3A3D42; }
QMenuBar::item:selected { background: #3A3D42; }
QMenu { background: #2A2D31; border: 1px solid #3A3D42; }
QMenu::item:selected { background: #3A3D42; }

QStatusBar { background: #2A2D31; color: #B8BCC2; }

QListWidget, QTableWidget, QGraphicsView {
    background: #25282C;
    border: 1px solid #3A3D42;
    selection-background-color: #3D5A7A;
    selection-color: #FFFFFF;
    alternate-background-color: #2A2D31;
}
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected { background: #3D5A7A; }

QHeaderView::section {
    background: #2A2D31;
    color: #E6E6E6;
    padding: 4px;
    border: 0px;
    border-right: 1px solid #3A3D42;
    border-bottom: 1px solid #3A3D42;
}
QTableWidget { gridline-color: #3A3D42; }
QTableCornerButton::section { background: #2A2D31; border: 1px solid #3A3D42; }

QLabel { color: #E6E6E6; background: transparent; }

QPushButton {
    background: #3A3D42;
    color: #E6E6E6;
    border: 1px solid #4A4D52;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover { background: #4A4D52; }
QPushButton:pressed { background: #2F3236; }

QSlider::groove:horizontal {
    height: 6px; background: #2A2D31; border-radius: 3px;
}
QSlider::sub-page:horizontal { background: #5A8DBC; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #E6E6E6; width: 14px; margin: -5px 0; border-radius: 7px;
}

QSplitter::handle { background: #3A3D42; }
QSplitter::handle:horizontal { width: 3px; }

QScrollBar:vertical, QScrollBar:horizontal {
    background: #25282C; border: 0; width: 10px; height: 10px;
}
QScrollBar::handle { background: #4A4D52; border-radius: 5px; min-height: 30px; min-width: 30px; }
QScrollBar::handle:hover { background: #5A5D62; }
QScrollBar::add-line, QScrollBar::sub-line { background: none; border: 0; }
"""


def main():
    """애플리케이션 진입점 — Qt 앱을 생성하고 메인 창을 띄운다."""
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
