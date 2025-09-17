from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

# Optional deps flags
_HAS_TORCH = _HAS_PIQ = _HAS_PYIQA = _HAS_PYWT = False

try:
    import torch
    import torchvision.transforms as T
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
    import pywt  # Wavelet
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

# ---------- Base sharpness metrics ----------
def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def wavelet_energy(gray: np.ndarray, wavelet: str = "db2", level: int = 2) -> float:
    if not _HAS_PYWT:
        raise RuntimeError("PyWavelets not installed: wavelet metric unavailable")
    x = gray.astype(np.float32)
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)
    energy = 0.0
    for lvl in range(1, len(coeffs)):
        cH, cV, cD = coeffs[lvl]
        energy += float(np.var(cH)) + float(np.var(cV)) + float(np.var(cD))
    return energy

# ---------- Multiscale variants (size-robust) ----------
def multiscale_laplacian_variance(gray: np.ndarray, scales=(1.0, 0.75, 0.5), weights=None) -> float:
    if weights is None:
        weights = np.ones(len(scales), dtype=np.float32)
    weights = weights / weights.sum()
    vals = []
    for s in scales:
        if s != 1.0:
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            g = cv2.resize(gray, (int(gray.shape[1]*s), int(gray.shape[0]*s)), interpolation=interp)
        else:
            g = gray
        vals.append(cv2.Laplacian(g, cv2.CV_64F).var())
    return float(np.dot(vals, weights))

def multiscale_wavelet_energy(gray: np.ndarray, scales=(1.0, 0.75, 0.5), weights=None, wavelet="db2", level=2) -> float:
    if not _HAS_PYWT:
        raise RuntimeError("PyWavelets not installed")
    if weights is None:
        weights = np.ones(len(scales), dtype=np.float32)
    weights = weights / weights.sum()
    vals = []
    for s in scales:
        if s != 1.0:
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            g = cv2.resize(gray, (int(gray.shape[1]*s), int(gray.shape[0]*s)), interpolation=interp)
        else:
            g = gray
        coeffs = pywt.wavedec2(g.astype(np.float32), wavelet=wavelet, level=level)
        e = 0.0
        for lvl in range(1, len(coeffs)):
            cH, cV, cD = coeffs[lvl]
            e += float(np.var(cH)) + float(np.var(cV)) + float(np.var(cD))
        vals.append(e)
    return float(np.dot(vals, weights))

# ---------- BRISQUE ----------
def _to_torch_tensor_rgb01(pil_img: Image.Image):
    x = pil_img.convert("RGB")
    x = np.array(x).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    return torch.from_numpy(x)

def brisque_score(pil_img: Image.Image) -> float:
    if not (_HAS_TORCH and _HAS_PIQ):
        raise RuntimeError("piq or torch not installed: BRISQUE unavailable")
    with torch.no_grad():
        x = _to_torch_tensor_rgb01(pil_img)
        return float(piq.brisque(x, data_range=1.0))

# ---------- NIQE ----------
class NIQEWrapper:
    def __init__(self):
        self.metric = None
        self.device = "cpu"
    def ready(self) -> bool:
        return self.metric is not None
    def try_init(self):
        if not (_HAS_TORCH and _HAS_PYIQA):
            return
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.metric = pyiqa.create_metric("niqe", device=self.device)
        except Exception:
            self.metric = None
    def score(self, pil_img: Image.Image) -> float:
        if self.metric is None:
            raise RuntimeError("pyiqa NIQE metric not available/failed to init")
        with torch.no_grad():
            x = T.ToTensor()(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
            v = self.metric(x)
        return float(v)

_niqe_singleton = NIQEWrapper()

def niqe_score(pil_img: Image.Image) -> float:
    if _niqe_singleton.metric is None:
        _niqe_singleton.try_init()
    return _niqe_singleton.score(pil_img)

# ---------- Size-invariant high-frequency energy ratio ----------
def high_freq_energy_ratio(gray: np.ndarray, band=(0.25, 1.0)) -> float:
    """
    정규화 주파수 r = sqrt(u^2+v^2) / Nyquist 로 밴드(예: 0.25~1.0) 내 파워 비율.
    해상도와 무관한 상대 대역 비교가 가능하여 블러 변화에 안정적으로 반응.
    """
    x = gray.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(x))
    P = np.abs(F) ** 2

    H, W = gray.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    ry = (yy - cy) / (H / 2.0)
    rx = (xx - cx) / (W / 2.0)
    r = np.sqrt(rx**2 + ry**2)  # 0..1

    rmin, rmax = band
    mask_band = (r >= rmin) & (r <= rmax)
    hf = P[mask_band].sum()
    total = P.sum() + 1e-12
    return float(hf / total)
