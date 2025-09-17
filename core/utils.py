from __future__ import annotations
import numpy as np
from PIL import Image
import cv2

def to_gray_np(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 grayscale numpy array [H,W]."""
    return np.array(pil_img.convert("L"), dtype=np.uint8)

def safe_norm01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] with epsilon guard."""
    x = x.astype("float32")
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return (x - mn) * 0.0
    return (x - mn) / (mx - mn + 1e-12)

def resize_for_eval_bounded(img_bgr_or_gray: np.ndarray, short_side: int = 720) -> np.ndarray:
    """
    짧은 변을 short_side로 맞추는 리사이즈.
    다운스케일 시 aliasing 방지를 위해 Gaussian 프리블러 적용.
    """
    if img_bgr_or_gray.ndim == 2:
        h, w = img_bgr_or_gray.shape
    else:
        h, w = img_bgr_or_gray.shape[:2]

    s = min(h, w)
    if s == short_side:
        return img_bgr_or_gray

    scale = short_side / float(s)

    # 다운스케일이면 프리블러(경험식)
    img = img_bgr_or_gray
    if scale < 1.0:
        sigma = max(0.5, 0.6 * (1.0/scale - 1.0))
        k = int(np.ceil(sigma * 6)) | 1
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    out = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return out

def pil_center_crop_256(pil_img: Image.Image) -> Image.Image:
    """
    NIQE/BRISQUE용 사이즈 표준화를 위해 짧은 변 256으로 맞춘 뒤 중앙 256x256 크롭.
    원본이 더 작으면 패딩 없이 리사이즈만 수행(크롭 생략).
    """
    im = pil_img.convert("RGB")
    w, h = im.size
    s = min(w, h)
    if s == 0:
        return im
    scale = 256.0 / float(s)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    im = im.resize((new_w, new_h), Image.BICUBIC)
    if new_w >= 256 and new_h >= 256:
        left = (new_w - 256) // 2
        top = (new_h - 256) // 2
        im = im.crop((left, top, left + 256, top + 256))
    return im
