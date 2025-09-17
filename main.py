import sys, traceback
import numpy as np
import cv2
from PIL import Image, ImageQt
from PySide6 import QtCore, QtGui, QtWidgets

# ---- Optional deps ----
_HAS_TORCH = _HAS_PIQ = _HAS_PYIQA = _HAS_PYWT = False
try:
    import torch, torchvision.transforms as T
    _HAS_TORCH = True
    try:
        import piq  # BRISQUE
        _HAS_PIQ = True
    except Exception:
        _HAS_PIQ = False
except Exception:
    _HAS_TORCH = _HAS_PIQ = False

try:
    import pyiqa  # NIQE
    _HAS_PYIQA = True
except Exception:
    _HAS_PYIQA = False

try:
    import pywt  # wavelet
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

# ---- Matplotlib embed ----
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ================= Metrics =================
def to_gray_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("L"), dtype=np.uint8)

def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def laplacian_map(gray: np.ndarray) -> np.ndarray:
    hp = cv2.Laplacian(gray, cv2.CV_64F)
    mag = np.abs(hp)
    mag = mag / (mag.max() + 1e-8) * 255.0
    return mag.astype(np.uint8)

def wavelet_energy(gray: np.ndarray, wavelet="db2", level=2) -> float:
    if not _HAS_PYWT:
        raise RuntimeError("pywt 미설치: Wavelet 지표 계산 불가")
    x = gray.astype(np.float32)
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)
    energy = 0.0
    for lvl in range(1, len(coeffs)):
        cH, cV, cD = coeffs[lvl]
        energy += float(np.var(cH)) + float(np.var(cV)) + float(np.var(cD))
    return energy

def wavelet_detail_map(gray: np.ndarray, wavelet="db2") -> np.ndarray:
    """1레벨 detail(|cH|+|cV|+|cD|)을 업샘플해 시각화"""
    if not _HAS_PYWT:
        return np.zeros_like(gray)
    x = gray.astype(np.float32)
    cA, (cH, cV, cD) = pywt.dwt2(x, wavelet=wavelet)
    det = np.abs(cH) + np.abs(cV) + np.abs(cD)
    det = det - det.min()
    det = det / (det.max() + 1e-8)
    det = cv2.resize(det, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (det * 255.0).astype(np.uint8)

def _to_torch_tensor_rgb01(pil_img: Image.Image):
    x = pil_img.convert("RGB")
    x = np.array(x).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # [3,H,W]
    x = np.expand_dims(x, 0)        # [1,3,H,W]
    return torch.from_numpy(x)

def brisque_score(pil_img: Image.Image) -> float:
    if not (_HAS_TORCH and _HAS_PIQ):
        raise RuntimeError("piq 또는 torch 미설치: BRISQUE 계산 불가")
    with torch.no_grad():
        x = _to_torch_tensor_rgb01(pil_img)
        return float(piq.brisque(x, data_range=1.0))

# --- MSCN (BRISQUE/NIQE 특징 시각화용) ---
def mscn_map(gray: np.ndarray, ksize: int = 7, sigma: float = 7/6) -> np.ndarray:
    x = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(x, (ksize, ksize), sigma)
    mu_sq = mu * mu
    sigma_map = cv2.GaussianBlur(x * x, (ksize, ksize), sigma) - mu_sq
    sigma_map = np.sqrt(np.abs(sigma_map))
    mscn = (x - mu) / (sigma_map + 1.0)  # 안정화
    # 시각화를 위해 0~255로 매핑 (중앙 0 → 128)
    vis = (np.clip((mscn * 0.5) + 0.5, 0, 1) * 255.0).astype(np.uint8)
    return vis

# ================= Plot Canvas =================
class BarCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2.8), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot_metrics(self, lap, wav, bri, niq, view_mode="raw"):
        self.ax.clear()
        labels = ["LaplacianVar (↑)", "WaveletEnergy (↑)", "BRISQUE (↓)", "NIQE (↓)"]
        if view_mode == "normalized":
            v_lap = np.log10(1.0 + max(lap, 0.0))
            v_wav = np.log10(1.0 + max(wav, 0.0))
            v_bri = 1.0 / (1.0 + max(bri, 0.0))
            v_niq = 1.0 / (1.0 + max(niq, 0.0))
            values = [v_lap, v_wav, v_bri, v_niq]
            title = "Metrics (normalized)"
        else:
            values = [lap, wav, bri, niq]
            title = "Metrics (raw)"
        self.ax.bar(labels, values)
        self.ax.set_title(title)
        self.ax.set_ylabel("score")
        self.ax.tick_params(axis='x', rotation=12)
        self.fig.canvas.draw_idle()

# ================= UI =================
class ImageQualityMVP(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sharpness & NR-IQA MVP (Laplacian / Wavelet / BRISQUE / NIQE)")
        self.resize(1160, 760)

        central = QtWidgets.QWidget(self); self.setCentralWidget(central)

        # Left: preview (원본)
        self.preview = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(520, 390)
        self.preview.setStyleSheet("background:#202020; color:#aaa; border:1px solid #444;")

        # Right top: controls + numbers + chart
        self.btn_open = QtWidgets.QPushButton("Open Image…  (O)")
        self.btn_recalc = QtWidgets.QPushButton("Recalculate  (R)")
        self.btn_recalc.setEnabled(False)
        self.path_edit = QtWidgets.QLineEdit(placeholderText="No image loaded"); self.path_edit.setReadOnly(True)

        form = QtWidgets.QFormLayout()
        self.lbl_lap = QtWidgets.QLabel("-")
        self.lbl_wav = QtWidgets.QLabel("-")
        self.lbl_bri = QtWidgets.QLabel("-")
        self.lbl_niq = QtWidgets.QLabel("-")
        form.addRow("Laplacian variance", self.lbl_lap)
        form.addRow("Wavelet energy", self.lbl_wav)
        form.addRow("BRISQUE (lower better)", self.lbl_bri)
        form.addRow("NIQE (lower better)", self.lbl_niq)

        self.mode_raw = QtWidgets.QRadioButton("Raw")
        self.mode_norm = QtWidgets.QRadioButton("Normalized"); self.mode_norm.setChecked(True)
        mode_group = QtWidgets.QHBoxLayout()
        mode_group.addWidget(QtWidgets.QLabel("Plot mode:"))
        mode_group.addWidget(self.mode_raw); mode_group.addWidget(self.mode_norm); mode_group.addStretch(1)

        self.canvas = BarCanvas(self)

        # Right bottom: 변환 시각화(우하단)
        self.view_label = QtWidgets.QLabel("Transform view:")
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Original", "Laplacian map", "Wavelet detail", "BRISQUE MSCN", "NIQE MSCN"])
        self.view_image = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.view_image.setMinimumSize(480, 320)
        self.view_image.setStyleSheet("background:#101010; border:1px solid #333; color:#aaa;")

        tip = QtWidgets.QLabel(
            "• Laplacian/Wavelet: 값 ↑ ⇒ 선명(고주파) 경향\n"
            "• BRISQUE/NIQE: 값 ↓ ⇒ 품질 양호(무참조)\n"
            "• Transform: Laplacian/Detail/MSCN을 우하단에서 시각화"
        ); tip.setStyleSheet("color:#666;")

        # Layouts
        right_top = QtWidgets.QVBoxLayout()
        right_top.addWidget(self.btn_open); right_top.addWidget(self.btn_recalc); right_top.addWidget(self.path_edit)
        right_top.addSpacing(6); right_top.addLayout(form); right_top.addSpacing(6)
        right_top.addLayout(mode_group); right_top.addWidget(self.canvas, stretch=1)

        right_bottom = QtWidgets.QVBoxLayout()
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.view_label); hb.addWidget(self.view_combo); hb.addStretch(1)
        right_bottom.addLayout(hb); right_bottom.addWidget(self.view_image)

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
        self._gray: np.ndarray | None = None
        self._last_values = dict(lap=np.nan, wav=np.nan, bri=np.nan, niq=np.nan)

        # NIQE metric (pyiqa) 1회 생성
        self._niqe_metric = None
        if _HAS_TORCH and _HAS_PYIQA:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._niqe_metric = pyiqa.create_metric('niqe', device=device)
            except Exception as e:
                print("[NIQE init error]", e)

        # Signals
        self.btn_open.clicked.connect(self.on_open)
        self.btn_recalc.clicked.connect(self.on_recalc)
        self.mode_raw.toggled.connect(self._update_plot_mode)
        self.view_combo.currentIndexChanged.connect(self._update_transform_view)

        # Shortcuts (QtWidgets에 존재)
        QtGui.QShortcut(QtGui.QKeySequence("O"), self, activated=self.on_open)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.on_recalc)

    # ---------- helpers ----------
    def _set_preview(self, pil_img: Image.Image | None):
        if pil_img is None:
            self.preview.setText("No Image"); return
        qimg = ImageQt.ImageQt(pil_img.convert("RGB"))
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _set_view_image_from_numpy(self, img: np.ndarray):
        if img.ndim == 2:
            show = Image.fromarray(img)
        else:
            show = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qimg = ImageQt.ImageQt(show)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.view_image.setPixmap(pix.scaled(self.view_image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        if self._pil_image is not None:
            self._set_preview(self._pil_image)
            self._update_transform_view()

    # ---------- actions ----------
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
        )
        if not path: return
        try:
            img = Image.open(path)
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Error", f"이미지 열기 실패:\n{ex}"); return

        self._pil_image = img
        self._gray = to_gray_np(img)
        self.path_edit.setText(path)
        self.btn_recalc.setEnabled(True)
        self._set_preview(img)
        self._compute_and_update()
        self._update_transform_view()

    def on_recalc(self):
        if self._pil_image is None: return
        self._compute_and_update()
        self._update_transform_view()

    # ---------- compute & plots ----------
    def _update_plot_mode(self):
        if any(np.isnan(list(self._last_values.values()))): return
        mode = "normalized" if self.mode_norm.isChecked() else "raw"
        self.canvas.plot_metrics(self._last_values["lap"], self._last_values["wav"],
                                 self._last_values["bri"], self._last_values["niq"],
                                 view_mode=mode)

    def _compute_and_update(self):
        try:
            pil = self._pil_image; gray = self._gray
            # Lap
            lap = laplacian_variance(gray); self.lbl_lap.setText(f"{lap:,.3f}")
            # Wave
            try:
                wav = wavelet_energy(gray, wavelet="db2", level=2); self.lbl_wav.setText(f"{wav:,.3f}")
            except Exception as ex_w:
                wav = np.nan; self.lbl_wav.setText("N/A"); print("[Wavelet error]", ex_w, traceback.format_exc())
            # BRISQUE
            try:
                bri = brisque_score(pil); self.lbl_bri.setText(f"{bri:.3f}")
            except Exception as ex_b:
                bri = np.nan; self.lbl_bri.setText("N/A"); print("[BRISQUE error]", ex_b, traceback.format_exc())
            # NIQE (pyiqa, cached metric)
            try:
                if self._niqe_metric is None: raise RuntimeError("pyiqa NIQE metric 초기화 실패/미설치")
                with torch.no_grad():
                    x = T.ToTensor()(pil.convert('RGB')).unsqueeze(0).to(next(self._niqe_metric.parameters()).device)
                    niq = float(self._niqe_metric(x))
                self.lbl_niq.setText(f"{niq:.3f}")
            except Exception as ex_n:
                niq = np.nan; self.lbl_niq.setText("N/A"); print("[NIQE error]", ex_n, traceback.format_exc())

            self._last_values = dict(lap=lap, wav=wav, bri=bri, niq=niq)
            self._update_plot_mode()

        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Error", f"지표 계산 실패:\n{ex}")
            print(traceback.format_exc())

    # ---------- transform view ----------
    def _update_transform_view(self):
        if self._pil_image is None or self._gray is None:
            self.view_image.setText("No image"); return
        mode = self.view_combo.currentText()
        try:
            if mode == "Original":
                show = np.array(self._pil_image.convert("RGB"))[:, :, ::-1]  # BGR for cv2-like
                self._set_view_image_from_numpy(show)
            elif mode == "Laplacian map":
                self._set_view_image_from_numpy(laplacian_map(self._gray))
            elif mode == "Wavelet detail":
                self._set_view_image_from_numpy(wavelet_detail_map(self._gray, wavelet="db2"))
            elif mode == "BRISQUE MSCN":
                self._set_view_image_from_numpy(mscn_map(self._gray))
            elif mode == "NIQE MSCN":
                self._set_view_image_from_numpy(mscn_map(self._gray))  # 동일 MSCN 기반, NIQE 관점 시각화
            else:
                self.view_image.setText("N/A")
        except Exception as e:
            self.view_image.setText(f"Transform error: {e}")
            print("[Transform error]", e)
# ------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ImageQualityMVP()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
