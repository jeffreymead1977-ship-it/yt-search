"""
Embedded PyVista point cloud viewer.

Tools:
  polygon   — click to place lasso vertices; orbit still works between clicks;
              double-click or Enter to close; Escape to cancel.
  limit_box — interactive axis-aligned clip box (PyVista box widget).
  wand      — click a seed point; KDTree radius-grow selects neighbours.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from ui.polygon_overlay import PolygonOverlay

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkRenderingCore import vtkPointPicker
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

from models.tank_scan import SegmentationResult, Label, LABEL_COLORS

_SEL_RGB     = np.array([255, 230, 40],  dtype=np.uint8)   # selection highlight
_CLICK_TOL   = 5    # pixels — below this is a "click", above is a "drag/orbit"
_DBLCLICK_MS = 400  # max ms between two clicks to count as double-click


class ViewerWidget(QWidget):
    selection_changed = Signal(int)   # count of currently selected points

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── core state ────────────────────────────────────────────────────
        self._pts:      Optional[np.ndarray] = None    # (N,3) float64
        self._labels:   Optional[np.ndarray] = None    # (N,)  int8
        self._selected: Optional[np.ndarray] = None    # (N,)  bool
        self._clip_mask:Optional[np.ndarray] = None    # (N,)  bool  (None = all visible)
        self._visible_indices: Optional[np.ndarray] = None
        self._undo_stack: list[np.ndarray] = []
        self._cloud_actor = None

        # ── tool state ────────────────────────────────────────────────────
        self._tool = "none"                  # "polygon" | "limit_box" | "wand"
        self._polygon_verts: list[tuple[int,int]] = []
        self._press_pos: Optional[tuple[int,int]] = None
        self._last_click_time = 0            # ms timestamp
        self._press_obs  = None
        self._release_obs = None
        self._key_obs    = None
        self._wand_radius = 0.3              # metres, configurable
        self._wand_same_label = True
        self._kdtree = None                  # lazy-built scipy KDTree
        self._clip_active = False

        if _PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background("#0f1117")
            layout.addWidget(self._plotter.interactor)
            self._overlay = PolygonOverlay(self)
            self._overlay.raise_()
            self._plotter.interactor.installEventFilter(self)
            self._show_placeholder()
        else:
            self._plotter = None
            self._overlay = None
            layout.addWidget(QLabel(
                "pyvistaqt not installed.\npip install pyvista pyvistaqt",
                alignment=Qt.AlignCenter,
            ))

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def display_result(self, result: SegmentationResult) -> None:
        if not self._plotter:
            return
        self._pts      = result.points.astype(np.float64)
        self._labels   = result.labels.copy()
        self._selected = np.zeros(len(self._pts), dtype=bool)
        self._clip_mask = None
        self._undo_stack.clear()
        self._kdtree = None
        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()
        self._rebuild_cloud(reset_camera=True)

    def set_tool(self, tool: str) -> None:
        """Switch active tool. Pass "none" to disable all tools."""
        if not self._plotter or self._pts is None:
            return
        previous = self._tool
        self._tool = tool

        # Tear down previous tool
        if previous == "polygon":
            self._remove_vtk_observers()
            self._polygon_verts.clear()
            if self._overlay:
                self._overlay.clear()
        elif previous == "limit_box":
            pass   # box widget persists until explicitly reset
        elif previous == "wand":
            self._remove_vtk_observers()

        # Set up new tool
        if tool == "polygon":
            self._install_vtk_observers()
        elif tool == "limit_box":
            self._activate_limit_box()
        elif tool == "wand":
            self._install_vtk_observers()

    def apply_label(self, label_value: int) -> None:
        if self._labels is None or self._selected is None:
            return
        if not self._selected.any():
            return
        self._undo_stack.append(self._labels.copy())
        self._labels[self._selected] = label_value
        self._selected[:] = False
        self._kdtree = None
        if self._overlay:
            self._overlay.clear()
        self._polygon_verts.clear()
        self._refresh_display()
        self.selection_changed.emit(0)

    def clear_selection(self) -> None:
        if self._selected is not None:
            self._selected[:] = False
        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()
        self._refresh_display()
        self.selection_changed.emit(0)

    def reset_limit_box(self) -> None:
        self._clip_mask = None
        self._clip_active = False
        if self._plotter:
            try:
                self._plotter.clear_box_widgets()
            except Exception:
                pass
        self._rebuild_cloud(reset_camera=False)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._labels = self._undo_stack.pop()
        if self._selected is not None:
            self._selected[:] = False
        self._kdtree = None
        self._refresh_display()
        self.selection_changed.emit(0)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def set_wand_radius(self, radius_m: float) -> None:
        self._wand_radius = radius_m

    def set_wand_same_label(self, same_label: bool) -> None:
        self._wand_same_label = same_label

    # ══════════════════════════════════════════════════════════════════════
    # VTK observers (polygon + wand)
    # ══════════════════════════════════════════════════════════════════════

    def _install_vtk_observers(self) -> None:
        iren = self._plotter.iren
        self._press_obs   = iren.AddObserver("LeftButtonPressEvent",   self._on_lmb_press)
        self._release_obs = iren.AddObserver("LeftButtonReleaseEvent", self._on_lmb_release)
        self._key_obs     = iren.AddObserver("KeyPressEvent",          self._on_key_press)

    def _remove_vtk_observers(self) -> None:
        iren = self._plotter.iren
        for attr in ("_press_obs", "_release_obs", "_key_obs"):
            obs = getattr(self, attr, None)
            if obs is not None:
                try:
                    iren.RemoveObserver(obs)
                except Exception:
                    pass
            setattr(self, attr, None)

    def _on_lmb_press(self, caller, _event) -> None:
        self._press_pos = caller.GetEventPosition()

    def _on_lmb_release(self, caller, _event) -> None:
        if self._press_pos is None:
            return
        pos = caller.GetEventPosition()
        dx  = abs(pos[0] - self._press_pos[0])
        dy  = abs(pos[1] - self._press_pos[1])
        self._press_pos = None

        if dx > _CLICK_TOL or dy > _CLICK_TOL:
            return   # it was a drag/orbit — ignore

        if self._tool == "polygon":
            self._polygon_add_vertex(pos)
        elif self._tool == "wand":
            self._wand_pick(pos)

    def _on_key_press(self, caller, _event) -> None:
        key = caller.GetKeySym()
        if key in ("Return", "KP_Enter"):
            if self._tool == "polygon":
                self._polygon_close()
        elif key == "Escape":
            self._polygon_verts.clear()
            if self._overlay:
                self._overlay.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Qt event filter — double-click + mouse-move tracking for overlay
    # ══════════════════════════════════════════════════════════════════════

    def eventFilter(self, obj, event) -> bool:
        if obj is not self._plotter.interactor:
            return False

        etype = event.type()

        if etype == QEvent.MouseButtonDblClick and self._tool == "polygon":
            # Remove the last vertex (which was added on the first click of
            # the double-click) then close
            if self._polygon_verts:
                self._polygon_verts.pop()
            self._polygon_close()
            return True   # consume so VTK doesn't treat it as two clicks

        if etype == QEvent.MouseMove and self._tool == "polygon" and self._polygon_verts:
            p = event.position().toPoint()
            self._overlay.set_vertices(self._polygon_verts, (p.x(), p.y()))

        return False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._overlay:
            self._overlay.setGeometry(0, 0, self.width(), self.height())

    # ══════════════════════════════════════════════════════════════════════
    # Polygon lasso
    # ══════════════════════════════════════════════════════════════════════

    def _polygon_add_vertex(self, screen_pos: tuple[int, int]) -> None:
        self._polygon_verts.append(screen_pos)
        if self._overlay:
            self._overlay.set_vertices(self._polygon_verts)

    def _polygon_close(self) -> None:
        if len(self._polygon_verts) < 3:
            self._polygon_verts.clear()
            if self._overlay:
                self._overlay.clear()
            return

        vis_idx = self._visible_indices
        if vis_idx is None:
            vis_idx = np.arange(len(self._pts))

        vis_pts     = self._pts[vis_idx]
        disp, front = self._world_to_display(vis_pts)
        in_poly     = _pts_in_polygon(self._polygon_verts, disp)
        hit_vis     = front & in_poly

        # Map visible indices back to full array
        shift = self._plotter.iren.GetShiftKey()
        if shift:
            self._selected[vis_idx[hit_vis]] = True
        else:
            self._selected[:] = False
            self._selected[vis_idx[hit_vis]] = True

        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()
        self._refresh_display()
        self.selection_changed.emit(int(self._selected.sum()))

    # ══════════════════════════════════════════════════════════════════════
    # Limit box
    # ══════════════════════════════════════════════════════════════════════

    def _activate_limit_box(self) -> None:
        if self._pts is None:
            return
        self._clip_active = True
        pts = self._pts
        bounds = [
            float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()),
            float(pts[:, 2].min()), float(pts[:, 2].max()),
        ]
        self._plotter.add_box_widget(
            callback=self._on_box_changed,
            bounds=bounds,
            rotation_enabled=False,
        )

    def _on_box_changed(self, box, *_) -> None:
        b = box.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
        pts = self._pts
        self._clip_mask = (
            (pts[:, 0] >= b[0]) & (pts[:, 0] <= b[1]) &
            (pts[:, 1] >= b[2]) & (pts[:, 1] <= b[3]) &
            (pts[:, 2] >= b[4]) & (pts[:, 2] <= b[5])
        )
        self._refresh_display()

    # ══════════════════════════════════════════════════════════════════════
    # Magic wand
    # ══════════════════════════════════════════════════════════════════════

    def _wand_pick(self, screen_pos: tuple[int, int]) -> None:
        if self._pts is None:
            return

        # Find the closest rendered point using vtkPointPicker
        picker = vtkPointPicker()
        picker.SetTolerance(0.025)
        picked = picker.Pick(screen_pos[0], screen_pos[1], 0,
                             self._plotter.renderer)
        if not picked:
            return
        picked_id = picker.GetPointId()
        if picked_id < 0:
            return

        # Map from rendered (visible) index to full array index
        vis_idx = self._visible_indices
        if vis_idx is None or picked_id >= len(vis_idx):
            return
        seed_idx   = int(vis_idx[picked_id])
        seed_pt    = self._pts[seed_idx]
        seed_label = int(self._labels[seed_idx])

        # Build KDTree if needed (only over visible points)
        if self._kdtree is None:
            from scipy.spatial import KDTree
            self._kdtree = KDTree(self._pts[vis_idx])

        # Radius search
        hit_vis = np.array(
            self._kdtree.query_ball_point(seed_pt, self._wand_radius),
            dtype=np.intp,
        )
        if not len(hit_vis):
            return

        # Map back to full indices
        hit_full = vis_idx[hit_vis]

        # Optionally restrict to same current label
        if self._wand_same_label:
            hit_full = hit_full[self._labels[hit_full] == seed_label]

        shift = self._plotter.iren.GetShiftKey()
        if not shift:
            self._selected[:] = False
        self._selected[hit_full] = True

        self._refresh_display()
        self.selection_changed.emit(int(self._selected.sum()))

    # ══════════════════════════════════════════════════════════════════════
    # Rendering helpers
    # ══════════════════════════════════════════════════════════════════════

    def _color_array(self) -> np.ndarray:
        n      = len(self._pts)
        colors = np.zeros((n, 3), dtype=np.uint8)
        for label, rgb in LABEL_COLORS.items():
            colors[self._labels == int(label)] = rgb
        if self._selected is not None and self._selected.any():
            colors[self._selected] = _SEL_RGB
        return colors

    def _compute_visible(self) -> np.ndarray:
        if self._clip_mask is not None:
            return np.where(self._clip_mask)[0]
        return np.arange(len(self._pts))

    def _safe_cam(self):
        try:
            return self._plotter.camera_position
        except Exception:
            return None

    def _rebuild_cloud(self, reset_camera: bool = False) -> None:
        if not self._plotter or self._pts is None:
            return
        cam = None if reset_camera else self._safe_cam()
        if self._cloud_actor is not None:
            self._plotter.remove_actor(self._cloud_actor)
            self._cloud_actor = None

        vis_idx = self._compute_visible()
        self._visible_indices = vis_idx

        cloud = pv.PolyData(self._pts[vis_idx].astype(np.float32))
        cloud['colors'] = self._color_array()[vis_idx]
        self._cloud_actor = self._plotter.add_mesh(
            cloud, scalars='colors', rgb=True,
            point_size=2, render_points_as_spheres=False, lighting=False,
        )
        if cam:
            self._plotter.camera_position = cam
        else:
            self._plotter.reset_camera()
        self._plotter.render()

    def _refresh_display(self) -> None:
        """Update colors without touching box widget or camera."""
        self._rebuild_cloud(reset_camera=False)

    def _show_placeholder(self) -> None:
        self._plotter.clear()
        self._plotter.add_text(
            "Open an E57 file and click Segment\nto view the point cloud",
            position="center", font_size=14, color="#444444",
        )
        self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════
    # Projection helpers
    # ══════════════════════════════════════════════════════════════════════

    def _world_to_display(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project world-space points to display pixels.
        Returns (display_xy (N,2), in_front (N,bool)).
        """
        renderer = self._plotter.renderer
        camera   = renderer.GetActiveCamera()
        aspect   = renderer.GetTiledAspectRatio()
        mvp      = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
        M = np.array([[mvp.GetElement(i, j) for j in range(4)] for i in range(4)])

        n      = len(pts)
        pts_h  = np.ones((n, 4))
        pts_h[:, :3] = pts
        clip   = (M @ pts_h.T).T          # (N, 4)

        w        = clip[:, 3]
        in_front = w > 1e-6
        ndc      = np.zeros((n, 2))
        ndc[in_front] = clip[in_front, :2] / w[in_front, None]

        vp      = renderer.GetViewport()
        win_w, win_h = renderer.GetRenderWindow().GetSize()
        vp_w    = (vp[2] - vp[0]) * win_w
        vp_h    = (vp[3] - vp[1]) * win_h
        vp_x    = vp[0] * win_w
        vp_y    = vp[1] * win_h

        disp    = np.zeros((n, 2))
        disp[:, 0] = (ndc[:, 0] + 1) * 0.5 * vp_w + vp_x
        disp[:, 1] = (ndc[:, 1] + 1) * 0.5 * vp_h + vp_y
        return disp, in_front

    # ── Cleanup ────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:
        if self._plotter:
            self._plotter.close()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Pure-numpy point-in-polygon  (ray-casting, no external deps)
# ══════════════════════════════════════════════════════════════════════════

def _pts_in_polygon(
    polygon: list[tuple[int, int]],
    points:  np.ndarray,           # (N, 2)
) -> np.ndarray:
    poly  = np.asarray(polygon, dtype=np.float64)
    px, py = points[:, 0], points[:, 1]
    n_v    = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j = n_v - 1
    for i in range(n_v):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross  = ((yi > py) != (yj > py)) & \
                 (px < xi + (xj - xi) * (py - yi) / (yj - yi + 1e-10))
        inside ^= cross
        j = i
    return inside
