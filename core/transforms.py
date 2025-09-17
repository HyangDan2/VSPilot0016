from __future__ import annotations
import numpy as np
import cv2

try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

from .utils import safe_norm01

def laplacian_map(gray: np.ndarray) -> np.ndarray:
    hp = cv2.Laplacian(gray, cv2.CV_64F)
    mag = np.abs(hp)
    vis = (safe_norm01(mag) * 255.0).astype(np.uint8)
    return vis

def wavelet_detail_map(gray: np.ndarray, wavelet: str = "db2") -> np.ndarray:
    if not _HAS_PYWT:
        return np.zeros_like(gray)
    x = gray.astype(np.float32)
    cA, (cH, cV, cD) = pywt.dwt2(x, wavelet=wavelet)
    det = np.abs(cH) + np.abs(cV) + np.abs(cD)
    det = safe_norm01(det)
    det = cv2.resize(det, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (det * 255.0).astype(np.uint8)

def mscn_map(gray: np.ndarray, ksize: int = 7, sigma: float = 7/6) -> np.ndarray:
    x = gray.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(x, (ksize, ksize), sigma)
    sigma_map = cv2.GaussianBlur(x * x, (ksize, ksize), sigma) - mu * mu
    sigma_map = np.sqrt(np.abs(sigma_map))
    mscn = (x - mu) / (sigma_map + 1.0)
    vis = np.clip((mscn * 0.5) + 0.5, 0, 1)
    return (vis * 255.0).astype(np.uint8)

def spectrum_log_map(gray: np.ndarray) -> np.ndarray:
    """
    2D 파워 스펙트럼의 로그 맵(시각화용). 중앙(DC) → 저주파, 외곽 → 고주파.
    """
    x = gray.astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(x))
    P = np.log1p(np.abs(F))
    vis = safe_norm01(P)
    return (vis * 255.0).astype(np.uint8)
