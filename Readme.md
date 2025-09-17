# Image Quality MVP

This project is a PySide6-based **No-Reference Image Quality Assessment (NR-IQA) MVP**, supporting:

- **Sharpness metrics**: Laplacian variance, Wavelet energy
- **NR-IQA metrics**: BRISQUE, NIQE
- **Transform visualizations**: Laplacian map, Wavelet detail, MSCN map
- **Interactive GUI** with charts and side-by-side transform previews

## Features
- Load image (PNG, JPG, BMP, TIFF)
- Compute 4 metrics automatically
- Plot raw/normalized scores
- Show transform maps in lower-right panel
- Keyboard shortcuts:  
  - **O** = Open Image  
  - **R** = Recalculate  

## Installation
```bash
git clone https://github.com/yourname/image_quality_mvp.git
cd image_quality_mvp
pip install -r requirements.txt
```

## Usage
```bash
python app/main.py
```

## Requirements
``` bash
Python 3.10+
PySide6
OpenCV
Pillow
Matplotlib
torch, piq (optional, for BRISQUE)
pyiqa (optional, for NIQE)
PyWavelets (optional, for Wavelet energy)
```

## License
MIT — see [LICENSE](./LICENSE)
