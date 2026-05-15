"""Embedded PyVista point cloud viewer.

Performance features
--------------------
- Eye-Dome Lighting (EDL) — screen-space depth shader.
- Octree LOD — spatially adaptive; close geometry = fine detail, far = coarse.
- Adaptive point size — fewer points → larger points.
- Camera-driven rendering — 150 ms debounce after movement ends.

Edit tools
----------
- polygon   — click vertices, double-click / Enter to close.
- limit_box — interactive axis-aligned clip box.
- wand      — KDTree radius-grow from a clicked seed point.

Measurement tools
-----------------
- distance  — pick 2 points → Euclidean distance.
- diameter  — pick N points + Enter → cylinder fit.
- area      — pick N points + Enter → plane fit + convex hull.

Panorama
--------
- show_panorama / hide_panorama — textured sphere inside the cloud.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from ui.polygon_overlay import PolygonOverlay
from processing.octree_lod import (
    OctreeNode, build_octree, collect_indices,
    get_frustum_planes, adaptive_point_size,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkRenderingCore import vtkPointPicker
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False

from models.tank_scan import SegmentationResult, Label, LABEL_COLORS

_SEL_RGB         = np.array([255, 230, 40], dtype=np.uint8)
_CLICK_TOL       = 5      # pixels — below → click, above → drag/orbit
_LOD_DEBOUNCE_MS = 150    # ms after camera stops before LOD refresh

# Quick uniform subsample for instant preview before octree is ready
def _quick_sample(pts: np.ndarray, target: int = 200_000) -> np.ndarray:
    n    = len(pts)
    step = max(1, n // target)
    return np.arange(0, n, step, dtype=np.intp)


class ViewerWidget(QWidget):
    selection_changed  = Signal(int)     # count of selected points
    lod_info_changed   = Signal(str)     # e.g. "Octree — 1.2 M pts"
    measurement_done   = Signal(object)  # MeasurementResult

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

        # ── octree LOD state ──────────────────────────────────────────────
        self._octree: Optional[OctreeNode] = None
        self._lod_thread: Optional[QThread] = None
        self._lod_worker = None
        self._edl_enabled = True
        self._lod_threshold_px: float = 80.0
        self._max_pts: int = 4_000_000
        self._point_size_override: Optional[float] = None
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

        # ── measurement state ─────────────────────────────────────────────
        self._measure_tool: str = "none"
        self._measure_picks: list[np.ndarray] = []
        self._measure_actors: list = []

        # ── panorama state ────────────────────────────────────────────────
        self._pano_actor = None

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
    # Public API — data
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
        self._octree = None
        self._polygon_verts.clear()
        if self._overlay:
            self._overlay.clear()

        self._visible_indices = _quick_sample(self._pts, 200_000)
        self._render_indices(self._visible_indices, reset_camera=True)
        self.lod_info_changed.emit("Building octree…")

        self._start_lod_build()

    # ══════════════════════════════════════════════════════════════════════
    # Public API — edit tools
    # ══════════════════════════════════════════════════════════════════════

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
        self._clip_mask   = None
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

    # ══════════════════════════════════════════════════════════════════════
    # Public API — rendering tunables
    # ══════════════════════════════════════════════════════════════════════

    def set_edl(self, enabled: bool) -> None:
        self._edl_enabled = enabled
        if not self._plotter:
            return
        if enabled:
            self._plotter.enable_eye_dome_lighting()
        else:
            self._plotter.disable_eye_dome_lighting()
        self._plotter.render()

    def set_lod_threshold(self, px: int) -> None:
        self._lod_threshold_px = float(px)
        self._full_lod_refresh()

    def set_max_pts(self, n: int) -> None:
        self._max_pts = n
        self._full_lod_refresh()

    def set_background_color(self, hex_color: str) -> None:
        if self._plotter:
            self._plotter.set_background(hex_color)
            self._plotter.render()

    def set_point_size_override(self, v: Optional[float]) -> None:
        self._point_size_override = v
        self._refresh_display()

    def set_force_lod(self, level: Optional[int]) -> None:
        """Legacy compatibility — ignored; octree uses spatial threshold."""
        pass

    def set_wand_radius(self, r: float) -> None:
        self._wand_radius = r

    def set_wand_same_label(self, v: bool) -> None:
        self._wand_same_label = v

    # ══════════════════════════════════════════════════════════════════════
    # Public API — panorama
    # ══════════════════════════════════════════════════════════════════════

    def show_panorama(self, pano, opacity: float = 1.0) -> None:
        """Display a textured panoramic sphere. pano is a PanoramaData."""
        if not self._plotter:
            return
        self.hide_panorama()
        try:
            from processing.panorama import make_panorama_sphere
            sphere, texture = make_panorama_sphere(pano)
            self._pano_actor = self._plotter.add_mesh(
                sphere,
                texture=texture,
                opacity=opacity,
                lighting=False,
            )
            self._plotter.render()
        except Exception:
            pass

    def hide_panorama(self) -> None:
        if self._pano_actor is not None and self._plotter:
            try:
                self._plotter.remove_actor(self._pano_actor)
            except Exception:
                pass
            self._pano_actor = None
            self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════
    # Public API — measurements
    # ══════════════════════════════════════════════════════════════════════

    def set_measure_tool(self, tool: str) -> None:
        """Set active measurement tool: "distance", "diameter", "area", "none"."""
        self._measure_tool = tool
        self._measure_picks.clear()
        if tool != "none":
            self._install_vtk_observers()
        else:
            if self._tool == "none":
                self._remove_vtk_observers()

    def clear_measurements(self) -> None:
        if self._plotter:
            for actor in self._measure_actors:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
        self._measure_actors.clear()
        self._measure_picks.clear()
        if self._plotter:
            self._plotter.render()

    # ══════════════════════════════════════════════════════════════════════
    # LOD build
    # ══════════════════════════════════════════════════════════════════════

    def _start_lod_build(self) -> None:
        if self._pts is None:
            return
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

    def _on_lod_built(self, octree: OctreeNode) -> None:
        self._octree = octree
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
        self._lod_timer.stop()
        self._lod_timer.start(_LOD_DEBOUNCE_MS)

    def _on_lod_timer(self) -> None:
        self._full_lod_refresh()

    # ══════════════════════════════════════════════════════════════════════
    # VTK observers (polygon + wand + measurements)
    # ══════════════════════════════════════════════════════════════════════

    def _install_vtk_observers(self) -> None:
        if self._press_obs is not None:
            return  # already installed
        iren = self._plotter.iren
        self._press_obs   = iren.AddObserver("LeftButtonPressEvent",   self._on_lmb_press)
        self._release_obs = iren.AddObserver("LeftButtonReleaseEvent", self._on_lmb_release)
        self._key_obs     = iren.AddObserver("KeyPressEvent",          self._on_key_press)

    def _remove_vtk_observers(self) -> None:
        if self._measure_tool != "none":
            return  # keep observers for measurement
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
        if (abs(pos[0] - self._press_pos[0]) > _CLICK_TOL or
                abs(pos[1] - self._press_pos[1]) > _CLICK_TOL):
            self._press_pos = None
            return   # drag/orbit
        self._press_pos = None

        if self._measure_tool != "none":
            self._measure_pick(pos)
        elif self._tool == "polygon":
            self._polygon_add_vertex(pos)
        elif self._tool == "wand":
            self._wand_pick(pos)

    def _on_key_press(self, caller, _event) -> None:
        key = caller.GetKeySym()
        if key in ("Return", "KP_Enter"):
            if self._measure_tool in ("diameter", "area"):
                self._measure_finish()
            elif self._tool == "polygon":
                self._polygon_close()
        elif key == "Escape":
            self._polygon_verts.clear()
            self._measure_picks.clear()
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
    # Measurement picking
    # ══════════════════════════════════════════════════════════════════════

    def _measure_pick(self, screen_pos: tuple[int, int]) -> None:
        """Pick a 3D world point and accumulate for the active measurement tool."""
        if not self._plotter:
            return
        picker = vtkPointPicker()
        picker.SetTolerance(0.025)
        picked = picker.Pick(screen_pos[0], screen_pos[1], 0, self._plotter.renderer)
        if not picked:
            return
        world_pt = np.array(picker.GetPickPosition(), dtype=np.float64)

        self._measure_picks.append(world_pt)

        if self._measure_tool == "distance" and len(self._measure_picks) >= 2:
            self._measure_finish()

    def _measure_finish(self) -> None:
        """Compute and emit measurement from accumulated picks."""
        picks = self._measure_picks
        tool  = self._measure_tool

        if tool == "distance" and len(picks) >= 2:
            from processing.measure import measure_distance
            result = measure_distance(picks[0], picks[1])
        elif tool == "diameter" and len(picks) >= 3:
            from processing.measure import measure_diameter
            pts = np.array(picks)
            result = measure_diameter(pts)
        elif tool == "area" and len(picks) >= 3:
            from processing.measure import measure_area
            pts = np.array(picks)
            result = measure_area(pts)
        else:
            return

        self._measure_picks.clear()
        self._draw_measurement(result)
        self.measurement_done.emit(result)

    def _draw_measurement(self, result) -> None:
        """Draw annotation geometry for a measurement result."""
        if not self._plotter or not result.annotation_pts:
            return
        pts = result.annotation_pts
        try:
            if len(pts) == 2:
                line = pv.Line(pts[0], pts[1])
                actor = self._plotter.add_mesh(
                    line, color="yellow", line_width=2, lighting=False
                )
                self._measure_actors.append(actor)
            elif len(pts) >= 3:
                arr = np.array(pts)
                cloud = pv.PolyData(arr.astype(np.float32))
                actor = self._plotter.add_mesh(
                    cloud, color="yellow", point_size=8,
                    render_points_as_spheres=True, lighting=False
                )
                self._measure_actors.append(actor)
            self._plotter.render()
        except Exception:
            pass

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
        """Pick octree indices from camera + frustum and re-render."""
        if not self._plotter or self._pts is None:
            return

        if self._octree is not None:
            try:
                cam_pos = np.array(self._plotter.camera_position[0], dtype=np.float64)
            except Exception:
                cam_pos = self._octree.center + np.array([0, 0, self._octree.half_diag * 3])

            try:
                frustum = get_frustum_planes(self._plotter.renderer)
            except Exception:
                frustum = np.zeros((6, 4))
                frustum[:, 3] = 1e9  # degenerate: accept everything

            win_h = float(self._plotter.renderer.GetRenderWindow().GetSize()[1])
            if win_h < 1:
                win_h = 600.0

            fov_deg = self._plotter.renderer.GetActiveCamera().GetViewAngle()
            fov_rad = math.radians(fov_deg)
            fov_half_tan = math.tan(fov_rad / 2.0)

            idx = collect_indices(
                self._octree,
                cam_pos,
                frustum,
                win_h,
                fov_half_tan,
                threshold_px=self._lod_threshold_px,
                max_pts=self._max_pts,
            )

            if self._clip_mask is not None and len(idx) > 0:
                idx = idx[self._clip_mask[idx]]

            n_shown = len(idx)
            self.lod_info_changed.emit(f"Octree LOD — {n_shown:,} pts")
        else:
            idx = _quick_sample(self._pts, 200_000)
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

        if self._point_size_override is not None:
            pt_size = self._point_size_override
        else:
            pt_size = adaptive_point_size(len(idx))

        self._cloud_actor = self._plotter.add_mesh(
            cloud,
            scalars='colors',
            rgb=True,
            point_size=pt_size,
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

        disp        = np.zeros((n, 2))
        disp[:, 0]  = (ndc[:, 0] + 1) * 0.5 * vp_w + vp_x
        disp[:, 1]  = (ndc[:, 1] + 1) * 0.5 * vp_h + vp_y
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

def _pts_in_polygon(polygon: list[tuple[int, int]], points: np.ndarray) -> np.ndarray:
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
