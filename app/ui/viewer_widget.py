"""Embedded PyVista point cloud viewer with rubber-band edit mode."""
from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkInteractionStyle import (
        vtkInteractorStyleRubberBandPick,
        vtkInteractorStyleTrackballCamera,
    )
    from vtkmodules.vtkRenderingCore import vtkAreaPicker
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

from models.tank_scan import SegmentationResult, Label, LABEL_COLORS

# Colour used to highlight the active selection
_SELECTION_RGB = (255, 255, 80)


class ViewerWidget(QWidget):
    selection_changed = Signal(int)   # emits count of selected points

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── state ──────────────────────────────────────────────────────────
        self._pts:      Optional[np.ndarray] = None   # (N, 3) float64
        self._labels:   Optional[np.ndarray] = None   # (N,)   int8
        self._selected: Optional[np.ndarray] = None   # (N,)   bool
        self._cloud:    Optional[pv.PolyData] = None
        self._undo_stack: list[np.ndarray]   = []     # copies of label arrays
        self._edit_mode = False
        self._press_pos: Optional[tuple[int, int]] = None
        self._press_obs  = None
        self._release_obs = None

        if _PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background("#0f1117")
            layout.addWidget(self._plotter.interactor)
            self._show_placeholder()
        else:
            self._plotter = None
            layout.addWidget(QLabel(
                "pyvistaqt not installed — 3D viewer unavailable.\n"
                "pip install pyvista pyvistaqt",
                alignment=Qt.AlignCenter,
            ))

    # ── Public API ─────────────────────────────────────────────────────────

    def display_result(self, result: SegmentationResult) -> None:
        if not self._plotter:
            return
        self._pts     = result.points.astype(np.float64)
        self._labels  = result.labels.copy()
        self._selected = np.zeros(len(self._pts), dtype=bool)
        self._undo_stack.clear()
        self._cloud = pv.PolyData(self._pts.astype(np.float32))
        self._rebuild_cloud(reset_camera=True)

    def set_edit_mode(self, active: bool) -> None:
        if not self._plotter or self._pts is None:
            return
        self._edit_mode = active
        if active:
            self._enable_rubber_band()
        else:
            self._disable_rubber_band()
            self._selected[:] = False
            self._refresh_display()
            self.selection_changed.emit(0)

    def apply_label(self, label_value: int) -> None:
        """Relabel currently selected points and push to undo stack."""
        if self._labels is None or self._selected is None:
            return
        if not self._selected.any():
            return
        self._undo_stack.append(self._labels.copy())
        self._labels[self._selected] = label_value
        self._selected[:] = False
        self._refresh_display()
        self.selection_changed.emit(0)

    def clear_selection(self) -> None:
        if self._selected is not None:
            self._selected[:] = False
            self._refresh_display()
            self.selection_changed.emit(0)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._labels = self._undo_stack.pop()
        if self._selected is not None:
            self._selected[:] = False
        self._refresh_display()
        self.selection_changed.emit(0)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    # ── Internal rendering ─────────────────────────────────────────────────

    def _color_array(self) -> np.ndarray:
        """Build per-point uint8 RGB, with selected points highlighted."""
        n = len(self._pts)
        colors = np.zeros((n, 3), dtype=np.uint8)
        for label, rgb in LABEL_COLORS.items():
            colors[self._labels == int(label)] = rgb
        if self._selected is not None and self._selected.any():
            colors[self._selected] = _SELECTION_RGB
        return colors

    def _rebuild_cloud(self, reset_camera: bool = False) -> None:
        cam = None if reset_camera else self._plotter.camera_position
        self._plotter.clear()
        self._cloud['colors'] = self._color_array()
        self._plotter.add_mesh(
            self._cloud,
            scalars='colors',
            rgb=True,
            point_size=2,
            render_points_as_spheres=False,
            lighting=False,
        )
        if cam:
            self._plotter.camera_position = cam
        else:
            self._plotter.reset_camera()
        self._plotter.render()

    def _refresh_display(self) -> None:
        """Update colors without resetting the camera."""
        if self._cloud is None:
            return
        self._rebuild_cloud(reset_camera=False)

    def _show_placeholder(self) -> None:
        self._plotter.clear()
        self._plotter.add_text(
            "Open an E57 file and click Segment\nto view the point cloud",
            position="center",
            font_size=14,
            color="#444444",
        )
        self._plotter.render()

    # ── Edit mode interaction ──────────────────────────────────────────────

    def _enable_rubber_band(self) -> None:
        style = vtkInteractorStyleRubberBandPick()
        self._plotter.iren.SetInteractorStyle(style)
        self._press_obs   = self._plotter.iren.AddObserver(
            "LeftButtonPressEvent",   self._on_lmb_press)
        self._release_obs = self._plotter.iren.AddObserver(
            "LeftButtonReleaseEvent", self._on_lmb_release)

    def _disable_rubber_band(self) -> None:
        if self._press_obs is not None:
            self._plotter.iren.RemoveObserver(self._press_obs)
            self._press_obs = None
        if self._release_obs is not None:
            self._plotter.iren.RemoveObserver(self._release_obs)
            self._release_obs = None
        style = vtkInteractorStyleTrackballCamera()
        self._plotter.iren.SetInteractorStyle(style)

    def _on_lmb_press(self, caller, _event) -> None:
        self._press_pos = caller.GetEventPosition()

    def _on_lmb_release(self, caller, _event) -> None:
        release_pos = caller.GetEventPosition()
        if self._press_pos is None or self._pts is None:
            return
        self._do_pick(self._press_pos, release_pos,
                      add=bool(caller.GetShiftKey()))
        self._press_pos = None

    def _do_pick(
        self,
        pos1: tuple[int, int],
        pos2: tuple[int, int],
        add: bool = False,
    ) -> None:
        x0 = min(pos1[0], pos2[0])
        y0 = min(pos1[1], pos2[1])
        x1 = max(pos1[0], pos2[0])
        y1 = max(pos1[1], pos2[1])

        # Single click → expand to 8 px area so it still picks nearby points
        if x1 - x0 < 4:
            x0 -= 4; x1 += 4
        if y1 - y0 < 4:
            y0 -= 4; y1 += 4

        picker = vtkAreaPicker()
        picker.AreaPick(int(x0), int(y0), int(x1), int(y1),
                        self._plotter.renderer)

        frustum = picker.GetFrustum()
        if frustum is None:
            return

        # Test all points against the 6 frustum planes (pure numpy)
        pts    = self._pts
        inside = np.ones(len(pts), dtype=bool)
        for i in range(frustum.GetNumberOfPlanes()):
            plane  = frustum.GetPlane(i)
            normal = np.array(plane.GetNormal())
            origin = np.array(plane.GetOrigin())
            inside &= ((pts - origin) @ normal) >= 0

        if add:
            self._selected |= inside
        else:
            self._selected = inside.copy()

        self._refresh_display()
        self.selection_changed.emit(int(self._selected.sum()))

    # ── Cleanup ────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._plotter:
            self._plotter.close()
        super().closeEvent(event)
