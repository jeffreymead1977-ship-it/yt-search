"""Embedded PyVista point cloud viewer."""
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

from models.tank_scan import SegmentationResult, LABEL_COLORS


class ViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background("#0f1117")
            layout.addWidget(self._plotter.interactor)
            self._show_placeholder()
        else:
            layout.addWidget(QLabel(
                "pyvistaqt not installed — 3D viewer unavailable.\n"
                "Run: pip install pyvista pyvistaqt",
                alignment=Qt.AlignCenter,
            ))
            self._plotter = None

    def _show_placeholder(self):
        if not self._plotter:
            return
        self._plotter.clear()
        self._plotter.add_text(
            "Open an E57 file and click Segment\nto view the point cloud",
            position="center",
            font_size=14,
            color="#444444",
        )
        self._plotter.reset_camera()

    def display_result(self, result: SegmentationResult) -> None:
        if not self._plotter:
            return

        self._plotter.clear()

        pts    = result.points
        labels = result.labels

        # Build per-point RGB array
        colors = np.zeros((len(pts), 3), dtype=np.uint8)
        for label, rgb in LABEL_COLORS.items():
            colors[labels == int(label)] = rgb

        cloud = pv.PolyData(pts.astype(np.float32))
        cloud["colors"] = colors

        self._plotter.add_mesh(
            cloud,
            scalars="colors",
            rgb=True,
            point_size=2,
            render_points_as_spheres=False,
            lighting=False,
        )
        self._plotter.reset_camera()

    def closeEvent(self, event):
        if self._plotter:
            self._plotter.close()
        super().closeEvent(event)
