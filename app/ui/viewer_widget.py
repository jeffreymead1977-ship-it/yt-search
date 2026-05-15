"""
Embedded PyVista point cloud viewer.

Performance features
--------------------
- Eye-Dome Lighting (EDL) — screen-space depth shader; makes sparse LOD
  look as good as full density.  One call to enable.
- Multi-level LOD — pre-built at load time; at render time the appropriate
  level (0 finest → 4 coarsest) is chosen from camera distance.
- Adaptive point size — fewer points on screen → larger points, so the
  screen always looks filled.
- Camera-driven rendering — only re-render when the camera moves; a 150 ms
  debounce after movement ends triggers a full LOD refresh.  During
  interaction the current cloud stays frozen so the frame rate stays high.

Edit tools
----------
- polygon   — click vertices, orbit between them, double-click / Enter to
              close; numpy ray-casting selection in screen space.
- limit_box — interactive axis-aligned clip box; hides points outside.
- wand      — KDTree radius-grow from a clicked seed point.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from ui.polygon_overlay import PolygonOverlay
from processing.lod import PointCloudLOD, quick_sample, adaptive_point_size

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkRenderingCore import vtkPointPicker
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

from models.tank_scan import SegmentationResult, Label, LABEL_COLORS

_SEL_RGB   = np.array([255, 230, 40], dtype=np.uint8)
_CLICK_TOL = 5      # pixels — below → click, above → drag/orbit
_LOD_DEBOUNCE_MS = 150   # ms after camera stops before LOD refresh


class ViewerWidget(QWidget):
    selection_changed  = Signal(int)    # count of selected points
    lod_info_changed   = Signal(str)    # e.g. "LOD 2 — 1.2 M pts"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── core state ────────────────────────────────────────────────────
        self._pts:       Optional[np.ndarray] = None
        self._labels:    Optional[np.ndarray] = None
        self._selected:  Optional[np.ndarray] = None
        self._clip_mask: Optional[np.ndarray] = None
        self._visible_indices: Optional[np.ndarray] = None
        self._undo_stack: list[np.ndarray] = []
        self._cloud_actor = None

        # ── LOD state ─────────────────────────────────────────────────────
        self._lod: Optional[PointCloudLOD] = None
        self._lod_thread: Optional[QThread] = None
        self._lod_worker = None
        self._edl_enabled = True
        self._force_lod:   Optional[int] = None
        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._on_lod_timer)
        self._cam_obs = None

        # ── tool state ────────────────────────────────────────────────────
        self._tool = "none"
        self._polygon_verts: list[tuple[int, int]] = []
        self._press_pos: Optional[tuple[int, int]] = None
        self._press_obs = self._release_obs = self._key_obs = None
        self._wand_radius     = 0.30
        self._wand_same_label = True
        self._kdtree          = None
        self._clip_active     = False

        if _PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background("#0f1117")
            if self._edl_enabled:
                self._plotter.enable_eye_dome_lighting()
            layout.addWidget(self._plotter.interactor)
            self._overlay = PolygonOverlay(self)
            self._overlay.raise_()
            self._plotter.interactor.installEventFilter(self)
            self._install_camera_observer()
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
        self._lod = None
        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()

        # Show a quick preview immediately, then build LOD in background
        self._visible_indices = quick_sample(self._pts, 200_000)
        self._render_indices(self._visible_indices, reset_camera=True)
        self.lod_info_changed.emit("Building LOD…")

        self._start_lod_build()

    def set_tool(self, tool: str) -> None:
        if not self._plotter or self._pts is None:
            return
        prev = self._tool
        self._tool = tool

        if prev == "polygon":
            self._remove_vtk_observers()
            self._polygon_verts.clear()
            if self._overlay:
                self._overlay.clear()
        elif prev == "wand":
            self._remove_vtk_observers()

        if tool in ("polygon", "wand"):
            self._install_vtk_observers()
        elif tool == "limit_box":
            self._activate_limit_box()

    def apply_label(self, label_value: int) -> None:
        if self._labels is None or not (self._selected is not None and self._selected.any()):
            return
        self._undo_stack.append(self._labels.copy())
        self._labels[self._selected] = label_value
        self._selected[:] = False
        self._kdtree = None
        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()
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
        self._clip_mask  = None
        self._clip_active = False
        try:
            self._plotter.clear_box_widgets()
        except Exception:
            pass
        self._full_lod_refresh()

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

    def set_edl(self, enabled: bool) -> None:
        self._edl_enabled = enabled
        if not self._plotter:
            return
        if enabled:
            self._plotter.enable_eye_dome_lighting()
        else:
            self._plotter.disable_eye_dome_lighting()
        self._plotter.render()

    def set_force_lod(self, level: Optional[int]) -> None:
        """Override automatic LOD level. None = automatic."""
        self._force_lod = level
        self._full_lod_refresh()

    def set_wand_radius(self, r: float) -> None:
        self._wand_radius = r

    def set_wand_same_label(self, v: bool) -> None:
        self._wand_same_label = v

    # ══════════════════════════════════════════════════════════════════════
    # LOD build
    # ══════════════════════════════════════════════════════════════════════

    def _start_lod_build(self) -> None:
        if self._pts is None:
            return
        # Abort previous build if running
        if self._lod_thread and self._lod_thread.isRunning():
            self._lod_thread.quit()
            self._lod_thread.wait(1000)

        from workers.lod_worker import LODWorker

        self._lod_thread = QThread(self)
        self._lod_worker = LODWorker(self._pts)
        self._lod_worker.moveToThread(self._lod_thread)
        self._lod_thread.started.connect(self._lod_worker.run)
        self._lod_worker.progress.connect(lambda m: self.lod_info_changed.emit(m))
        self._lod_worker.finished.connect(self._on_lod_built)
        self._lod_worker.error.connect(lambda e: self.lod_info_changed.emit(f"LOD error: {e}"))
        self._lod_worker.finished.connect(self._lod_thread.quit)
        self._lod_thread.start()

    def _on_lod_built(self, lod: PointCloudLOD) -> None:
        self._lod = lod
        self._lod_thread = None
        self._lod_worker = None
        self._full_lod_refresh()

    # ══════════════════════════════════════════════════════════════════════
    # Camera observer + debounce
    # ══════════════════════════════════════════════════════════════════════

    def _install_camera_observer(self) -> None:
        self._cam_obs = self._plotter.iren.AddObserver(
            "EndInteractionEvent", self._on_camera_moved
        )

    def _on_camera_moved(self, *_) -> None:
        # Debounce: re-render LOD 150 ms after camera stops
        self._lod_timer.stop()
        self._lod_timer.start(_LOD_DEBOUNCE_MS)

    def _on_lod_timer(self) -> None:
        self._full_lod_refresh()

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
        if abs(pos[0]-self._press_pos[0]) > _CLICK_TOL or \
           abs(pos[1]-self._press_pos[1]) > _CLICK_TOL:
            self._press_pos = None
            return   # it was a drag
        self._press_pos = None
        if self._tool == "polygon":
            self._polygon_add_vertex(pos)
        elif self._tool == "wand":
            self._wand_pick(pos)

    def _on_key_press(self, caller, _event) -> None:
        key = caller.GetKeySym()
        if key in ("Return", "KP_Enter") and self._tool == "polygon":
            self._polygon_close()
        elif key == "Escape":
            self._polygon_verts.clear()
            if self._overlay:
                self._overlay.clear()

    # ══════════════════════════════════════════════════════════════════════
    # Qt event filter
    # ══════════════════════════════════════════════════════════════════════

    def eventFilter(self, obj, event) -> bool:
        if obj is not self._plotter.interactor:
            return False
        etype = event.type()
        if etype == QEvent.MouseButtonDblClick and self._tool == "polygon":
            if self._polygon_verts:
                self._polygon_verts.pop()
            self._polygon_close()
            return True
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

    def _polygon_add_vertex(self, pos: tuple[int, int]) -> None:
        self._polygon_verts.append(pos)
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
        b = box.bounds
        p = self._pts
        self._clip_mask = (
            (p[:, 0] >= b[0]) & (p[:, 0] <= b[1]) &
            (p[:, 1] >= b[2]) & (p[:, 1] <= b[3]) &
            (p[:, 2] >= b[4]) & (p[:, 2] <= b[5])
        )
        self._kdtree = None
        self._refresh_display()

    # ══════════════════════════════════════════════════════════════════════
    # Magic wand
    # ══════════════════════════════════════════════════════════════════════

    def _wand_pick(self, screen_pos: tuple[int, int]) -> None:
        if self._pts is None:
            return
        picker = vtkPointPicker()
        picker.SetTolerance(0.025)
        if not picker.Pick(screen_pos[0], screen_pos[1], 0, self._plotter.renderer):
            return
        picked_id = picker.GetPointId()
        if picked_id < 0:
            return

        vis_idx = self._visible_indices
        if vis_idx is None or picked_id >= len(vis_idx):
            return
        seed_idx   = int(vis_idx[picked_id])
        seed_label = int(self._labels[seed_idx])

        if self._kdtree is None:
            from scipy.spatial import KDTree
            search_pts = self._pts if self._clip_mask is None \
                         else self._pts[self._clip_mask]
            self._kdtree_base = (
                np.arange(len(self._pts)) if self._clip_mask is None
                else np.where(self._clip_mask)[0]
            )
            self._kdtree = KDTree(search_pts)

        hits_local = np.array(
            self._kdtree.query_ball_point(self._pts[seed_idx], self._wand_radius),
            dtype=np.intp,
        )
        if not len(hits_local):
            return
        hit_full = self._kdtree_base[hits_local]
        if self._wand_same_label:
            hit_full = hit_full[self._labels[hit_full] == seed_label]

        shift = self._plotter.iren.GetShiftKey()
        if not shift:
            self._selected[:] = False
        self._selected[hit_full] = True
        self._refresh_display()
        self.selection_changed.emit(int(self._selected.sum()))

    # ══════════════════════════════════════════════════════════════════════
    # Rendering
    # ══════════════════════════════════════════════════════════════════════

    def _color_array(self) -> np.ndarray:
        n      = len(self._pts)
        colors = np.zeros((n, 3), dtype=np.uint8)
        for label, rgb in LABEL_COLORS.items():
            colors[self._labels == int(label)] = rgb
        if self._selected is not None and self._selected.any():
            colors[self._selected] = _SEL_RGB
        return colors

    def _full_lod_refresh(self) -> None:
        """Pick LOD level from camera position and re-render."""
        if not self._plotter or self._pts is None:
            return

        if self._lod is not None:
            try:
                cam = np.array(self._plotter.camera_position[0])
            except Exception:
                cam = self._lod.center + np.array([0, 0, self._lod.radius * 3])
            idx, lv = self._lod.select(
                cam,
                clip_mask=self._clip_mask,
                force_level=self._force_lod,
            )
            n_shown = len(idx)
            level_names = ["Full res", "15 mm", "50 mm", "150 mm", "500 mm"]
            name = level_names[min(lv, len(level_names)-1)]
            self.lod_info_changed.emit(f"{name} — {n_shown:,} pts")
        else:
            idx = quick_sample(self._pts, 200_000)
            if self._clip_mask is not None:
                idx = idx[self._clip_mask[idx]]
            self.lod_info_changed.emit(f"Preview — {len(idx):,} pts")

        self._visible_indices = idx
        self._render_indices(idx, reset_camera=False)

    def _refresh_display(self) -> None:
        """Refresh colours without changing LOD or camera."""
        if self._visible_indices is not None:
            self._render_indices(self._visible_indices, reset_camera=False)

    def _render_indices(self, idx: np.ndarray, reset_camera: bool = False) -> None:
        cam = None if reset_camera else self._safe_cam()

        if self._cloud_actor is not None:
            self._plotter.remove_actor(self._cloud_actor)
            self._cloud_actor = None

        if len(idx) == 0:
            self._plotter.render()
            return

        pts_vis    = self._pts[idx].astype(np.float32)
        colors_vis = self._color_array()[idx]

        cloud = pv.PolyData(pts_vis)
        cloud['colors'] = colors_vis

        self._cloud_actor = self._plotter.add_mesh(
            cloud,
            scalars='colors',
            rgb=True,
            point_size=adaptive_point_size(len(idx)),
            render_points_as_spheres=False,
            lighting=False,
        )
        if cam:
            self._plotter.camera_position = cam
        else:
            self._plotter.reset_camera()
        self._plotter.render()

    def _safe_cam(self):
        try:
            return self._plotter.camera_position
        except Exception:
            return None

    def _show_placeholder(self) -> None:
        self._plotter.clear()
        self._plotter.add_text(
            "Open an E57 file and click Segment\nto view the point cloud",
            position="center", font_size=14, color="#444444",
        )
        self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════
    # Screen-space projection helpers
    # ══════════════════════════════════════════════════════════════════════

    def _world_to_display(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        renderer = self._plotter.renderer
        camera   = renderer.GetActiveCamera()
        aspect   = renderer.GetTiledAspectRatio()
        mvp      = camera.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
        M = np.array([[mvp.GetElement(i, j) for j in range(4)] for i in range(4)])

        n     = len(pts)
        pts_h = np.ones((n, 4))
        pts_h[:, :3] = pts
        clip  = (M @ pts_h.T).T

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
        self._lod_timer.stop()
        if self._lod_thread and self._lod_thread.isRunning():
            self._lod_thread.quit()
            self._lod_thread.wait(2000)
        if self._plotter:
            self._plotter.close()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════
# Pure-numpy point-in-polygon (ray-casting, no external deps)
# ══════════════════════════════════════════════════════════════════════════

def _pts_in_polygon(polygon: list[tuple[int,int]], points: np.ndarray) -> np.ndarray:
    poly   = np.asarray(polygon, dtype=np.float64)
    px, py = points[:, 0], points[:, 1]
    n_v    = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j      = n_v - 1
    for i in range(n_v):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross  = ((yi > py) != (yj > py)) & \
                 (px < xi + (xj - xi) * (py - yi) / (yj - yi + 1e-10))
        inside ^= cross
        j = i
    return inside
