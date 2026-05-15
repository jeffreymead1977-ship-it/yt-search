"""Left-hand control, results, and editing panel."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QDoubleSpinBox, QFormLayout, QFrame,
    QSizePolicy, QSpacerItem, QScrollArea, QCheckBox, QButtonGroup,
    QSlider, QListWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from models.tank_scan import Label, LABEL_COLORS, LABEL_DISPLAY_NAMES


class _Swatch(QFrame):
    def __init__(self, rgb: tuple[int, int, int], size: int = 12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        r, g, b = rgb
        self.setStyleSheet(
            f"background:{QColor(r,g,b).name()};border-radius:{size//3}px;"
        )


class Sidebar(QWidget):
    open_requested            = Signal()
    segment_requested         = Signal(str, dict)
    tool_activated            = Signal(str)       # "polygon"|"limit_box"|"wand"|"none"
    apply_label_requested     = Signal(int)
    clear_selection_requested = Signal()
    reset_box_requested       = Signal()
    undo_requested            = Signal()
    wand_radius_changed       = Signal(float)
    wand_same_label_changed   = Signal(bool)
    edl_toggled               = Signal(bool)
    lod_override_changed      = Signal(object)    # kept for compatibility
    export_ply_requested      = Signal()
    export_json_requested     = Signal()

    # Rendering tunables
    lod_threshold_changed     = Signal(int)       # px
    max_pts_changed           = Signal(int)
    point_size_changed        = Signal(object)    # float or None
    bg_color_changed          = Signal(str)       # hex string

    # Panorama
    pano_show_requested       = Signal(int, float)  # scan_index, opacity
    pano_hide_requested       = Signal()

    # Measurements
    measure_tool_changed      = Signal(str)
    clear_measurements_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(296)
        self._active_tool         = "none"
        self._active_measure_tool = "none"
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        root  = QVBoxLayout(inner)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_file_group())
        root.addWidget(self._build_scan_type_group())
        root.addWidget(self._build_options_group())
        root.addWidget(self._build_segment_btn())
        root.addWidget(self._build_results_group())
        root.addWidget(self._build_rendering_group())
        root.addWidget(self._build_panorama_group())
        root.addWidget(self._build_measure_group())
        root.addWidget(self._build_edit_group())
        root.addWidget(self._build_legend_group())
        root.addWidget(self._build_export_group())
        root.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Section builders ───────────────────────────────────────────────────

    def _build_file_group(self) -> QGroupBox:
        g = QGroupBox("Scan File")
        l = QVBoxLayout(g)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color:#888;font-size:11px;")
        l.addWidget(self._file_label)
        self._open_btn = QPushButton("Open E57…")
        self._open_btn.clicked.connect(self.open_requested)
        l.addWidget(self._open_btn)
        return g

    def _build_scan_type_group(self) -> QGroupBox:
        g = QGroupBox("Scan Type")
        l = QVBoxLayout(g)
        self._scan_combo = QComboBox()
        self._scan_combo.addItems(["External", "Internal"])
        l.addWidget(self._scan_combo)
        note = QLabel(
            "External: shell, out-of-roundness, roof profile.\n"
            "Internal: floor plate, deadwood, heating coils."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#666;font-size:10px;")
        l.addWidget(note)
        return g

    def _build_options_group(self) -> QGroupBox:
        g = QGroupBox("Processing Options")
        l = QFormLayout(g)
        l.setSpacing(5)
        self._voxel_spin = _spin(0.005, 0.1,  0.005, 0.01,  " m")
        self._plane_spin = _spin(0.005, 0.05, 0.005, 0.015, " m")
        self._cyl_spin   = _spin(0.005, 0.10, 0.005, 0.02,  " m")
        l.addRow("Voxel size:",         self._voxel_spin)
        l.addRow("Plane tolerance:",    self._plane_spin)
        l.addRow("Cylinder tolerance:", self._cyl_spin)
        return g

    def _build_segment_btn(self) -> QPushButton:
        self._segment_btn = QPushButton("Segment")
        self._segment_btn.setObjectName("primary")
        self._segment_btn.setEnabled(False)
        self._segment_btn.setFixedHeight(34)
        self._segment_btn.clicked.connect(self._on_segment)
        return self._segment_btn

    def _build_results_group(self) -> QGroupBox:
        g = QGroupBox("Results")
        l = QFormLayout(g)
        l.setSpacing(3)
        self._result_labels: dict[str, QLabel] = {}

        for key in ["Radius", "Diameter", "Height", "Floor Z", "Roof Z", "Total"]:
            lbl = _dim_lbl()
            l.addRow(f"{key}:", lbl)
            self._result_labels[key] = lbl

        for label in Label:
            lbl = _dim_lbl()
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(4)
            row_l.addWidget(_Swatch(LABEL_COLORS[label]))
            row_l.addWidget(QLabel(LABEL_DISPLAY_NAMES[label]))
            row_l.addStretch()
            l.addRow(row_w, lbl)
            self._result_labels[label.name] = lbl

        return g

    def _build_rendering_group(self) -> QGroupBox:
        g = QGroupBox("Rendering")
        l = QVBoxLayout(g)
        l.setSpacing(6)

        # LOD info label
        self._lod_label = QLabel("LOD: —")
        self._lod_label.setStyleSheet("color:#888;font-size:10px;")
        l.addWidget(self._lod_label)

        # EDL toggle
        self._edl_chk = QCheckBox("Eye-Dome Lighting (EDL)")
        self._edl_chk.setChecked(True)
        self._edl_chk.setToolTip(
            "Screen-space depth shader — makes sparse LOD look as dense as "
            "full resolution. Highly recommended."
        )
        self._edl_chk.toggled.connect(self.edl_toggled)
        l.addWidget(self._edl_chk)

        # LOD quality slider
        self._lod_quality_label = QLabel("Quality: 80 px — higher = more detail")
        self._lod_quality_label.setStyleSheet("color:#aaa;font-size:10px;")
        l.addWidget(self._lod_quality_label)
        self._lod_quality_slider = QSlider(Qt.Horizontal)
        self._lod_quality_slider.setRange(20, 200)
        self._lod_quality_slider.setValue(80)
        self._lod_quality_slider.setTickInterval(20)
        self._lod_quality_slider.valueChanged.connect(self._on_lod_quality_changed)
        l.addWidget(self._lod_quality_slider)

        # Max render pts
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Max pts:"))
        self._max_pts_combo = QComboBox()
        _max_pts_options = [
            ("500 K",  500_000),
            ("1 M",  1_000_000),
            ("2 M",  2_000_000),
            ("4 M",  4_000_000),
            ("8 M",  8_000_000),
        ]
        for label, value in _max_pts_options:
            self._max_pts_combo.addItem(label, value)
        self._max_pts_combo.setCurrentIndex(3)  # default 4 M
        self._max_pts_combo.currentIndexChanged.connect(self._on_max_pts_changed)
        row1.addWidget(self._max_pts_combo)
        l.addLayout(row1)

        # Point size
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Point size:"))
        self._pt_size_combo = QComboBox()
        _pt_size_options = [
            ("Auto", None),
            ("1",    1.0),
            ("1.5",  1.5),
            ("2",    2.0),
            ("3",    3.0),
            ("4",    4.0),
            ("5",    5.0),
        ]
        for label, value in _pt_size_options:
            self._pt_size_combo.addItem(label, value)
        self._pt_size_combo.setCurrentIndex(0)
        self._pt_size_combo.currentIndexChanged.connect(self._on_pt_size_changed)
        row2.addWidget(self._pt_size_combo)
        l.addLayout(row2)

        # Background
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Background:"))
        self._bg_combo = QComboBox()
        _bg_options = [
            ("Dark (default)", "#0f1117"),
            ("Black",          "#000000"),
            ("Grey",           "#444444"),
            ("White",          "#ffffff"),
        ]
        for label, value in _bg_options:
            self._bg_combo.addItem(label, value)
        self._bg_combo.setCurrentIndex(0)
        self._bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        row3.addWidget(self._bg_combo)
        l.addLayout(row3)

        return g

    def _build_panorama_group(self) -> QGroupBox:
        g = QGroupBox("Panorama")
        l = QVBoxLayout(g)
        l.setSpacing(6)

        self._pano_count_label = QLabel("0 panoramas found")
        self._pano_count_label.setStyleSheet("color:#888;font-size:10px;")
        l.addWidget(self._pano_count_label)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Scan:"))
        self._pano_combo = QComboBox()
        row1.addWidget(self._pano_combo)
        l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Opacity:"))
        self._pano_opacity_slider = QSlider(Qt.Horizontal)
        self._pano_opacity_slider.setRange(10, 100)
        self._pano_opacity_slider.setValue(100)
        row2.addWidget(self._pano_opacity_slider)
        l.addLayout(row2)

        btns = QHBoxLayout()
        self._pano_show_btn = QPushButton("Show")
        self._pano_hide_btn = QPushButton("Hide")
        self._pano_show_btn.clicked.connect(self._on_pano_show)
        self._pano_hide_btn.clicked.connect(self.pano_hide_requested)
        btns.addWidget(self._pano_show_btn)
        btns.addWidget(self._pano_hide_btn)
        l.addLayout(btns)

        return g

    def _build_measure_group(self) -> QGroupBox:
        g = QGroupBox("Measurements")
        l = QVBoxLayout(g)
        l.setSpacing(6)

        # Tool buttons
        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        self._measure_btns: dict[str, QPushButton] = {}
        self._measure_btn_group = QButtonGroup(self)
        self._measure_btn_group.setExclusive(False)

        for tool_id, label in [("distance", "Distance"), ("diameter", "Diameter"), ("area", "Area")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.toggled.connect(lambda checked, t=tool_id: self._on_measure_tool_toggled(t, checked))
            self._measure_btns[tool_id] = btn
            tool_row.addWidget(btn)

        l.addLayout(tool_row)

        hint = QLabel(
            "Distance: pick 2 pts  |  Diameter/Area: pick pts + Enter"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;font-size:10px;")
        l.addWidget(hint)

        # Results list
        self._measure_list = QListWidget()
        self._measure_list.setMaximumHeight(120)
        self._measure_list.setStyleSheet("font-size:11px;")
        l.addWidget(self._measure_list)

        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedHeight(26)
        clear_btn.clicked.connect(self._on_clear_measurements)
        l.addWidget(clear_btn)

        return g

    def _build_edit_group(self) -> QGroupBox:
        g = QGroupBox("Edit Tools")
        root = QVBoxLayout(g)
        root.setSpacing(8)

        # Tool palette
        palette = QHBoxLayout()
        palette.setSpacing(4)

        btn_defs = [
            ("polygon",   "Polygon",   "Click to place vertices\nDouble-click or Enter to close"),
            ("limit_box", "Limit Box", "Drag handles to clip the view\nPoints outside are hidden"),
            ("wand",      "Wand",      "Click a seed point to grow\na region-based selection"),
        ]
        self._tool_btns: dict[str, QPushButton] = {}
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)

        for tool_id, label, _ in btn_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.toggled.connect(lambda checked, t=tool_id: self._on_tool_toggled(t, checked))
            self._tool_btns[tool_id] = btn
            palette.addWidget(btn)

        root.addLayout(palette)

        # Hint label
        self._tool_hint = QLabel("")
        self._tool_hint.setWordWrap(True)
        self._tool_hint.setStyleSheet("color:#666;font-size:10px;")
        root.addWidget(self._tool_hint)

        # Selection info
        self._sel_label = QLabel("No points selected")
        self._sel_label.setStyleSheet("color:#aaa;font-size:11px;")
        root.addWidget(self._sel_label)

        # Wand options
        self._wand_widget = QWidget()
        wl = QFormLayout(self._wand_widget)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(4)
        self._wand_radius_spin = _spin(0.05, 10.0, 0.05, 0.30, " m")
        self._wand_radius_spin.valueChanged.connect(self.wand_radius_changed)
        self._wand_same_chk = QCheckBox("Same label only")
        self._wand_same_chk.setChecked(True)
        self._wand_same_chk.toggled.connect(self.wand_same_label_changed)
        wl.addRow("Radius:", self._wand_radius_spin)
        wl.addRow(self._wand_same_chk)
        self._wand_widget.setVisible(False)
        root.addWidget(self._wand_widget)

        # Limit box controls
        self._box_widget = QWidget()
        bl = QVBoxLayout(self._box_widget)
        bl.setContentsMargins(0, 0, 0, 0)
        self._reset_box_btn = QPushButton("Reset Box (show all)")
        self._reset_box_btn.clicked.connect(self._on_reset_box)
        bl.addWidget(self._reset_box_btn)
        self._box_widget.setVisible(False)
        root.addWidget(self._box_widget)

        # Label assignment buttons
        self._label_widget = QWidget()
        ll = QVBoxLayout(self._label_widget)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)

        self._label_btns: dict[int, QPushButton] = {}
        rows = [[Label.SHELL, Label.FLOOR], [Label.ROOF, Label.DEADWOOD], [Label.RUBBISH]]
        for row_labels in rows:
            row = QHBoxLayout()
            row.setSpacing(4)
            for lbl in row_labels:
                r, g_c, b = LABEL_COLORS[lbl]
                btn = QPushButton(LABEL_DISPLAY_NAMES[lbl])
                btn.setEnabled(False)
                btn.setFixedHeight(27)
                btn.setStyleSheet(
                    f"QPushButton{{"
                    f"background:rgba({r},{g_c},{b},55);"
                    f"border:1px solid rgba({r},{g_c},{b},130);"
                    f"border-radius:4px;color:#e0e0e0;}}"
                    f"QPushButton:hover{{background:rgba({r},{g_c},{b},100);}}"
                    f"QPushButton:disabled{{opacity:0.3;}}"
                )
                btn.clicked.connect(
                    lambda checked=False, v=int(lbl): self.apply_label_requested.emit(v)
                )
                self._label_btns[int(lbl)] = btn
                row.addWidget(btn)
            row.addStretch()
            ll.addLayout(row)

        # Clear + Undo
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)
        self._clear_btn = QPushButton("Clear Selection")
        self._undo_btn  = QPushButton("Undo  Ctrl+Z")
        for b in (self._clear_btn, self._undo_btn):
            b.setEnabled(False)
            b.setFixedHeight(27)
            ctrl.addWidget(b)
        self._clear_btn.clicked.connect(self.clear_selection_requested)
        self._undo_btn.clicked.connect(self.undo_requested)
        ll.addLayout(ctrl)

        self._label_widget.setVisible(False)
        root.addWidget(self._label_widget)

        return g

    def _build_legend_group(self) -> QGroupBox:
        g = QGroupBox("Legend")
        l = QVBoxLayout(g)
        l.setSpacing(3)
        for label in Label:
            row = QHBoxLayout()
            row.addWidget(_Swatch(LABEL_COLORS[label]))
            row.addWidget(QLabel(LABEL_DISPLAY_NAMES[label]))
            row.addStretch()
            l.addLayout(row)
        return g

    def _build_export_group(self) -> QGroupBox:
        g = QGroupBox("Export")
        l = QVBoxLayout(g)
        self._export_ply_btn  = QPushButton("Export Labeled PLY…")
        self._export_json_btn = QPushButton("Export Summary JSON…")
        for btn in (self._export_ply_btn, self._export_json_btn):
            btn.setEnabled(False)
            l.addWidget(btn)
        self._export_ply_btn.clicked.connect(self.export_ply_requested)
        self._export_json_btn.clicked.connect(self.export_json_requested)
        return g

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_segment(self):
        self.segment_requested.emit(
            self._scan_combo.currentText().lower(),
            {
                "voxel_size": self._voxel_spin.value(),
                "plane_dist": self._plane_spin.value(),
                "cyl_tol":    self._cyl_spin.value(),
            },
        )

    def _on_tool_toggled(self, tool_id: str, checked: bool) -> None:
        if checked:
            for tid, btn in self._tool_btns.items():
                if tid != tool_id:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self._active_tool = tool_id
        else:
            if self._active_tool == tool_id:
                self._active_tool = "none"

        self._update_tool_ui()
        self.tool_activated.emit(self._active_tool)

    def _on_reset_box(self):
        self.reset_box_requested.emit()

    def _on_lod_quality_changed(self, value: int) -> None:
        self._lod_quality_label.setText(f"Quality: {value} px — higher = more detail")
        self.lod_threshold_changed.emit(value)

    def _on_max_pts_changed(self, _idx: int) -> None:
        self.max_pts_changed.emit(self._max_pts_combo.currentData())

    def _on_pt_size_changed(self, _idx: int) -> None:
        self.point_size_changed.emit(self._pt_size_combo.currentData())

    def _on_bg_changed(self, _idx: int) -> None:
        self.bg_color_changed.emit(self._bg_combo.currentData())

    def _on_pano_show(self):
        idx = self._pano_combo.currentData()
        if idx is None:
            idx = 0
        opacity = self._pano_opacity_slider.value() / 100.0
        self.pano_show_requested.emit(idx, opacity)

    def _on_measure_tool_toggled(self, tool_id: str, checked: bool) -> None:
        if checked:
            for tid, btn in self._measure_btns.items():
                if tid != tool_id:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self._active_measure_tool = tool_id
        else:
            if self._active_measure_tool == tool_id:
                self._active_measure_tool = "none"
        self.measure_tool_changed.emit(self._active_measure_tool)

    def _on_clear_measurements(self):
        self._measure_list.clear()
        self.clear_measurements_requested.emit()

    # ── Tool UI visibility ─────────────────────────────────────────────────

    def _update_tool_ui(self):
        hints = {
            "polygon":   "Click to add vertices · Shift+click adds to selection\n"
                         "Double-click or Enter to finish · Escape to cancel\n"
                         "Drag to orbit as normal between clicks",
            "limit_box": "Drag the box handles to clip the visible cloud.\n"
                         "Points outside are hidden but not deleted.",
            "wand":      "Click any point to grow a radius selection.\n"
                         "Shift+click adds to existing selection.",
            "none":      "",
        }
        tool = self._active_tool
        self._tool_hint.setText(hints.get(tool, ""))
        self._wand_widget.setVisible(tool == "wand")
        self._box_widget.setVisible(tool == "limit_box")
        show_labels = tool in ("polygon", "wand")
        self._label_widget.setVisible(show_labels)
        self._sel_label.setVisible(tool in ("polygon", "wand"))

    # ── Public update methods ───────────────────────────────────────────────

    def set_file(self, path: str) -> None:
        import os
        self._file_label.setText(os.path.basename(path))
        self._segment_btn.setEnabled(True)

    def set_processing(self, active: bool) -> None:
        self._segment_btn.setEnabled(not active)
        self._open_btn.setEnabled(not active)
        self._export_ply_btn.setEnabled(False)
        self._export_json_btn.setEnabled(False)
        for btn in self._tool_btns.values():
            btn.setEnabled(not active)

    def set_result(self, result) -> None:
        cyl    = result.cylinder
        counts = result.point_counts()

        def _f(v, dp=2): return f"{v:.{dp}f} m" if v is not None else "—"

        self._result_labels["Radius"].setText(_f(cyl.radius_m   if cyl else None))
        self._result_labels["Diameter"].setText(_f(cyl.diameter_m if cyl else None))
        self._result_labels["Height"].setText(_f(cyl.height_m   if cyl else None))
        self._result_labels["Floor Z"].setText(_f(result.floor_z_m))
        self._result_labels["Roof Z"].setText(_f(result.roof_z_m))
        self._result_labels["Total"].setText(f"{len(result.points):,}")

        for label in Label:
            n   = counts.get(label.name, 0)
            pct = 100 * n / max(len(result.points), 1)
            self._result_labels[label.name].setText(f"{n:,}  ({pct:.1f}%)")

        self._segment_btn.setEnabled(True)
        self._export_ply_btn.setEnabled(True)
        self._export_json_btn.setEnabled(True)
        for btn in self._tool_btns.values():
            btn.setEnabled(True)
        for btn in self._measure_btns.values():
            btn.setEnabled(True)

    def set_lod_info(self, msg: str) -> None:
        self._lod_label.setText(f"Rendering: {msg}")

    def update_selection_count(self, count: int) -> None:
        if count == 0:
            self._sel_label.setText("No points selected")
        else:
            self._sel_label.setText(f"{count:,} points selected")
        has_sel = count > 0
        for btn in self._label_btns.values():
            btn.setEnabled(has_sel)
        self._clear_btn.setEnabled(True)

    def set_undo_available(self, available: bool) -> None:
        self._undo_btn.setEnabled(available)

    def set_pano_count(self, n: int) -> None:
        self._pano_count_label.setText(f"{n} panorama{'s' if n != 1 else ''} found")
        self._pano_combo.clear()
        for i in range(n):
            self._pano_combo.addItem(f"Scan {i}", i)

    def add_measurement(self, result) -> None:
        """Append a MeasurementResult to the list widget."""
        self._measure_list.addItem(result.label)
        self._measure_list.scrollToBottom()

    def set_measure_tool_enabled(self, enabled: bool) -> None:
        for btn in self._measure_btns.values():
            btn.setEnabled(enabled)


# ── Helpers ────────────────────────────────────────────────────────────────

def _spin(lo, hi, step, val, suffix) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    return s


def _dim_lbl() -> QLabel:
    l = QLabel("—")
    l.setStyleSheet("color:#888;")
    return l
