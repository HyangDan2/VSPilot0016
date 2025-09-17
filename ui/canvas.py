from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class BarCanvas(FigureCanvasQTAgg):
    """
    Bar chart to display Laplacian/Wavelet/BRISQUE/NIQE/HFER.
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2.8), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot_metrics(self, values, view_mode: str = "raw"):
        """
        values: dict with keys ['lap', 'wav', 'bri', 'niq', 'hfer']
        """
        self.ax.clear()
        labels = ["LaplacianVar (↑)", "WaveletEnergy (↑)", "BRISQUE (↓)", "NIQE (↓)", "HFER (↑)"]
        lap = values.get("lap", np.nan)
        wav = values.get("wav", np.nan)
        bri = values.get("bri", np.nan)
        niq = values.get("niq", np.nan)
        hfer = values.get("hfer", np.nan)

        if view_mode == "normalized":
            v_lap = np.log10(1.0 + max(0.0, 0 if np.isnan(lap) else lap))
            v_wav = np.log10(1.0 + max(0.0, 0 if np.isnan(wav) else wav))
            v_bri = 1.0 / (1.0 + (0 if np.isnan(bri) else max(bri, 0.0)))
            v_niq = 1.0 / (1.0 + (0 if np.isnan(niq) else max(niq, 0.0)))
            v_hfer = 0 if np.isnan(hfer) else float(hfer)  # 이미 0..1 비율
            plot_vals = [v_lap, v_wav, v_bri, v_niq, v_hfer]
            title = "Metrics (normalized)"
        else:
            plot_vals = [lap, wav, bri, niq, hfer]
            title = "Metrics (raw)"

        self.ax.bar(labels, plot_vals)
        self.ax.set_title(title)
        self.ax.set_ylabel("score")
        self.ax.tick_params(axis='x', rotation=12)
        self.fig.canvas.draw_idle()
