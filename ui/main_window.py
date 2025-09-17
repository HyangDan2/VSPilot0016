from __future__ import annotations
import traceback
import numpy as np
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui, QtWidgets

from core.utils import to_gray_np, resize_for_eval_bounded, pil_center_crop_256
from core.metrics import (
    laplacian_variance, wavelet_energy,
    multiscale_laplacian_variance, multiscale_wavelet_energy,
    brisque_score, niqe_score, high_freq_energy_ratio
)
from core.transforms import laplacian_map, wavelet_detail_map, mscn_map, spectrum_log_map
from ui.canvas import BarCanvas

class ImageQualityMVP(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sharpness & NR-IQA MVP (Size-Robust Mode + HFER)")
        self.resize(1240, 800)

        central = QtWidgets.QWidget(self); self.setCentralWidget(central)

        # Left: preview
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(560, 420)
        self.preview.setStyleSheet("background:#202020; color:#aaa; border:1px solid #444;")

        # Right top controls
        self.btn_open = QtWidgets.QPushButton("Open Image…  (O)")
        self.btn_recalc = QtWidgets.QPushButton("Recalculate  (R)")
        self.btn_recalc.setEnabled(False)
        self.path_edit = QtWidgets.QLineEdit(placeholderText="No image loaded")
        self.path_edit.setReadOnly(True)

        # Size-robust mode
        self.cb_size_robust = QtWidgets.QCheckBox("Size-Robust mode (resize+multiscale+HFER)")
        self.cb_size_robust.setChecked(True)

        form = QtWidgets.QFormLayout()
        self.lbl_lap = QtWidgets.QLabel("-")
        self.lbl_wav = QtWidgets.QLabel("-")
        self.lbl_bri = QtWidgets.QLabel("-")
        self.lbl_niq = QtWidgets.QLabel("-")
        self.lbl_hfer = QtWidgets.QLabel("-")
        form.addRow("Laplacian variance", self.lbl_lap)
        form.addRow("Wavelet energy", self.lbl_wav)
        form.addRow("BRISQUE (lower better)", self.lbl_bri)
        form.addRow("NIQE (lower better)", self.lbl_niq)
        form.addRow("HFER (high-freq energy ratio)", self.lbl_hfer)

        self.mode_raw = QtWidgets.QRadioButton("Raw")
        self.mode_norm = QtWidgets.QRadioButton("Normalized")
        self.mode_norm.setChecked(True)
        mode_group = QtWidgets.QHBoxLayout()
        mode_group.addWidget(QtWidgets.QLabel("Plot mode:"))
        mode_group.addWidget(self.mode_raw); mode_group.addWidget(self.mode_norm); mode_group.addStretch(1)

        self.canvas = BarCanvas(self)

        # Right bottom: transform preview
        self.view_label = QtWidgets.QLabel("Transform view:")
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems([
            "Original", "Laplacian map", "Wavelet detail", "BRISQUE MSCN", "NIQE MSCN", "Spectrum (log)"
        ])
        self.view_image = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.view_image.setMinimumSize(520, 340)
        self.view_image.setStyleSheet("background:#101010; border:1px solid #333; color:#aaa;")

        tip = QtWidgets.QLabel(
            "• Size-Robust: 입력 리사이즈(짧은 변=720) + 멀티스케일 집계 + HFER\n"
            "• Laplacian/Wavelet: 점수 ↑ ⇒ 더 날카로움\n"
            "• BRISQUE/NIQE: 점수 ↓ ⇒ 더 양호 (무참조)\n"
            "• HFER: 0~1 사이, 고주파 대역 에너지 비율 (블러 ↑ ⇒ HFER ↓)"
        )
        tip.setStyleSheet("color:#666;")

        # Layout
        right_top = QtWidgets.QVBoxLayout()
        right_top.addWidget(self.btn_open)
        right_top.addWidget(self.btn_recalc)
        right_top.addWidget(self.path_edit)
        right_top.addWidget(self.cb_size_robust)
        right_top.addSpacing(6)
        right_top.addLayout(form)
        right_top.addSpacing(6)
        right_top.addLayout(mode_group)
        right_top.addWidget(self.canvas, stretch=1)

        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.view_label)
        hb.addWidget(self.view_combo)
        hb.addStretch(1)

        right_bottom = QtWidgets.QVBoxLayout()
        right_bottom.addLayout(hb)
        right_bottom.addWidget(self.view_image)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(right_top, stretch=3)
        right.addSpacing(8)
        right.addLayout(right_bottom, stretch=2)
        right.addWidget(tip)

        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.preview, stretch=3)
        layout.addLayout(right, stretch=4)

        # State
        self._pil_image: Image.Image | None = None
        self._pil_image_eval: Image.Image | None = None  # size-robust 전처리 후
        self._gray: np.ndarray | None = None
        self._gray_eval: np.ndarray | None = None        # size-robust 전처리 후
        self._last_values = dict(lap=np.nan, wav=np.nan, bri=np.nan, niq=np.nan, hfer=np.nan)

        # Signals
        self.btn_open.clicked.connect(self.on_open)
        self.btn_recalc.clicked.connect(self.on_recalc)
        self.mode_raw.toggled.connect(self._update_plot_mode)
        self.view_combo.currentIndexChanged.connect(self._update_transform_view)
        self.cb_size_robust.toggled.connect(self._on_size_mode_toggle)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("O"), self, activated=self.on_open)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.on_recalc)

    # ---------- helpers ----------
    def _set_preview(self, pil_img: Image.Image | None):
        if pil_img is None:
            self.preview.setText("No Image")
            return
        qimg = ImageQt.ImageQt(pil_img.convert("RGB"))
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def _set_view_image_from_numpy(self, img: np.ndarray):
        if img.ndim == 2:
            show = Image.fromarray(img)
        else:
            import cv2
            show = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qimg = ImageQt.ImageQt(show)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.view_image.setPixmap(
            pix.scaled(self.view_image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        if self._pil_image_eval is not None:
            self._set_preview(self._pil_image_eval)
            self._update_transform_view()

    # ---------- actions ----------
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
        )
        if not path:
            return
        try:
            pil_orig = Image.open(path).convert("RGB")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open image:\n{ex}")
            return

        self.path_edit.setText(path)
        self._pil_image = pil_orig

        # Size-robust 전처리 (짧은 변=720) OR 원본 유지
        if self.cb_size_robust.isChecked():
            # PIL→np(BGR)로 변환하여 정규화 후 다시 PIL
            np_rgb = np.array(pil_orig)
            np_bgr = np_rgb[:, :, ::-1]
            np_bgr_norm = resize_for_eval_bounded(np_bgr, short_side=720)
            np_rgb_norm = np_bgr_norm[:, :, ::-1]
            pil_eval = Image.fromarray(np_rgb_norm)
        else:
            pil_eval = pil_orig

        self._pil_image_eval = pil_eval
        self._gray = to_gray_np(pil_orig)
        self._gray_eval = to_gray_np(pil_eval)

        self.btn_recalc.setEnabled(True)
        self._set_preview(pil_eval)
        self._compute_and_update()
        self._update_transform_view()

    def on_recalc(self):
        if self._pil_image is None:
            return
        # on_open 때 설정한 eval 버전을 다시 사용(토글 상태 유지)
        self._compute_and_update()
        self._update_transform_view()

    def _on_size_mode_toggle(self):
        # 모드 토글 시 즉시 재평가
        if self._pil_image is None:
            return
        pil_orig = self._pil_image
        if self.cb_size_robust.isChecked():
            np_rgb = np.array(pil_orig)
            np_bgr = np_rgb[:, :, ::-1]
            np_bgr_norm = resize_for_eval_bounded(np_bgr, short_side=720)
            np_rgb_norm = np_bgr_norm[:, :, ::-1]
            self._pil_image_eval = Image.fromarray(np_rgb_norm)
        else:
            self._pil_image_eval = pil_orig
        self._gray = to_gray_np(pil_orig)
        self._gray_eval = to_gray_np(self._pil_image_eval)
        self._set_preview(self._pil_image_eval)
        self._compute_and_update()
        self._update_transform_view()

    # ---------- compute & plots ----------
    def _update_plot_mode(self):
        if any(np.isnan(list(self._last_values.values()))):
            return
        mode = "normalized" if self.mode_norm.isChecked() else "raw"
        self.canvas.plot_metrics(self._last_values, view_mode=mode)

    def _compute_and_update(self):
        try:
            pil_eval = self._pil_image_eval
            gray_eval = self._gray_eval
            if pil_eval is None or gray_eval is None:
                return

            size_robust = self.cb_size_robust.isChecked()

            # Laplacian & Wavelet
            if size_robust:
                lap = multiscale_laplacian_variance(gray_eval, scales=(1.0, 0.75, 0.5))
                try:
                    wav = multiscale_wavelet_energy(gray_eval, scales=(1.0, 0.75, 0.5), wavelet="db2", level=2)
                except Exception as ex_w:
                    wav = np.nan
                    print("[Wavelet error]", ex_w, traceback.format_exc())
            else:
                lap = laplacian_variance(gray_eval)
                try:
                    wav = wavelet_energy(gray_eval, wavelet="db2", level=2)
                except Exception as ex_w:
                    wav = np.nan
                    print("[Wavelet error]", ex_w, traceback.format_exc())

            # BRISQUE / NIQE — size-robust일 때 256 center-crop 표준화
            try:
                pil_brisque = pil_center_crop_256(pil_eval) if size_robust else pil_eval
                bri = brisque_score(pil_brisque)
            except Exception as ex_b:
                bri = np.nan
                print("[BRISQUE error]", ex_b, traceback.format_exc())

            try:
                pil_niqe = pil_center_crop_256(pil_eval) if size_robust else pil_eval
                niq = niqe_score(pil_niqe)
            except Exception as ex_n:
                niq = np.nan
                print("[NIQE error]", ex_n, traceback.format_exc())

            # HFER (size-invariant high frequency energy ratio)
            try:
                hfer = high_freq_energy_ratio(gray_eval, band=(0.25, 1.0))
            except Exception as ex_h:
                hfer = np.nan
                print("[HFER error]", ex_h, traceback.format_exc())

            # Update labels
            self.lbl_lap.setText(f"{lap:,.3f}" if np.isfinite(lap) else "N/A")
            self.lbl_wav.setText(f"{wav:,.3f}" if np.isfinite(wav) else "N/A")
            self.lbl_bri.setText(f"{bri:.3f}" if np.isfinite(bri) else "N/A")
            self.lbl_niq.setText(f"{niq:.3f}" if np.isfinite(niq) else "N/A")
            self.lbl_hfer.setText(f"{hfer:.4f}" if np.isfinite(hfer) else "N/A")

            self._last_values = dict(lap=lap, wav=wav, bri=bri, niq=niq, hfer=hfer)
            self._update_plot_mode()

        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Error", f"Metric computation failed:\n{ex}")
            print(traceback.format_exc())

    # ---------- transform view ----------
    def _update_transform_view(self):
        if self._pil_image_eval is None or self._gray_eval is None:
            self.view_image.setText("No image")
            return
        mode = self.view_combo.currentText()
        try:
            if mode == "Original":
                show = np.array(self._pil_image_eval.convert("RGB"))[:, :, ::-1]  # BGR-like
                self._set_view_image_from_numpy(show)
            elif mode == "Laplacian map":
                self._set_view_image_from_numpy(laplacian_map(self._gray_eval))
            elif mode == "Wavelet detail":
                self._set_view_image_from_numpy(wavelet_detail_map(self._gray_eval, wavelet="db2"))
            elif mode == "BRISQUE MSCN":
                self._set_view_image_from_numpy(mscn_map(self._gray_eval))
            elif mode == "NIQE MSCN":
                self._set_view_image_from_numpy(mscn_map(self._gray_eval))
            elif mode == "Spectrum (log)":
                self._set_view_image_from_numpy(spectrum_log_map(self._gray_eval))
            else:
                self.view_image.setText("N/A")
        except Exception as e:
            self.view_image.setText(f"Transform error: {e}")
            print("[Transform error]", e)
