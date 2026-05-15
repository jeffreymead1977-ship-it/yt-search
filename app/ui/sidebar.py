"""Left-hand control, results, and manual-edit panel."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QDoubleSpinBox, QFormLayout, QFrame,
    QSizePolicy, QSpacerItem, QScrollArea,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

from models.tank_scan import Label, LABEL_COLORS, LABEL_DISPLAY_NAMES


class _Swatch(QFrame):
    def __init__(self, rgb: tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        r, g, b = rgb
        self.setStyleSheet(
            f"background:{QColor(r,g,b).name()};border-radius:2px;"
        )


class Sidebar(QWidget):
    open_requested        = Signal()
    segment_requested     = Signal(str, dict)   # scan_type, params
    edit_mode_toggled     = Signal(bool)
    apply_label_requested = Signal(int)          # Label value
    clear_selection_requested = Signal()
    undo_requested        = Signal()
    export_ply_requested  = Signal()
    export_json_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self._edit_active = False
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Wrap everything in a scroll area so it works on small screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_file_group())
        root.addWidget(self._build_scan_type_group())
        root.addWidget(self._build_options_group())
        root.addWidget(self._build_segment_btn())
        root.addWidget(self._build_results_group())
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
        note.setStyleSheet("color:#777;font-size:10px;")
        l.addWidget(note)
        return g

    def _build_options_group(self) -> QGroupBox:
        g = QGroupBox("Processing Options")
        l = QFormLayout(g)
        l.setSpacing(6)

        self._voxel_spin = _spin(0.005, 0.1, 0.005, 0.01, " m")
        self._plane_spin = _spin(0.005, 0.05, 0.005, 0.015, " m")
        self._cyl_spin   = _spin(0.005, 0.10, 0.005, 0.02, " m")

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
            lbl = _dim_label()
            l.addRow(f"{key}:", lbl)
            self._result_labels[key] = lbl

        for label in Label:
            lbl = _dim_label()
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(_Swatch(LABEL_COLORS[label]))
            row.addWidget(QLabel(LABEL_DISPLAY_NAMES[label]))
            l.addRow(row, lbl)
            self._result_labels[label.name] = lbl

        return g

    def _build_edit_group(self) -> QGroupBox:
        g = QGroupBox("Manual Edit")
        l = QVBoxLayout(g)
        l.setSpacing(6)

        # Edit mode toggle
        self._edit_toggle = QPushButton("Enter Edit Mode")
        self._edit_toggle.setCheckable(True)
        self._edit_toggle.setEnabled(False)
        self._edit_toggle.toggled.connect(self._on_edit_toggle)
        l.addWidget(self._edit_toggle)

        # Hint
        self._edit_hint = QLabel(
            "Drag to select  ·  Shift+drag adds to selection"
        )
        self._edit_hint.setWordWrap(True)
        self._edit_hint.setStyleSheet("color:#666;font-size:10px;")
        self._edit_hint.setVisible(False)
        l.addWidget(self._edit_hint)

        # Selection count
        self._sel_count = QLabel("No points selected")
        self._sel_count.setStyleSheet("color:#aaa;font-size:11px;")
        self._sel_count.setVisible(False)
        l.addWidget(self._sel_count)

        # Label assignment buttons (2 columns)
        self._label_btns: dict[int, QPushButton] = {}
        grid_rows = [
            [Label.SHELL,    Label.FLOOR],
            [Label.ROOF,     Label.DEADWOOD],
            [Label.RUBBISH,  None],
        ]
        for row_labels in grid_rows:
            row = QHBoxLayout()
            row.setSpacing(4)
            for lbl in row_labels:
                if lbl is None:
                    row.addStretch()
                    continue
                r, g_c, b = LABEL_COLORS[lbl]
                btn = QPushButton(LABEL_DISPLAY_NAMES[lbl])
                btn.setEnabled(False)
                btn.setFixedHeight(28)
                # Tinted background matching label colour
                btn.setStyleSheet(
                    f"QPushButton {{"
                    f"  background-color: rgba({r},{g_c},{b},60);"
                    f"  border: 1px solid rgba({r},{g_c},{b},140);"
                    f"  border-radius: 4px; color: #e0e0e0;"
                    f"}}"
                    f"QPushButton:hover {{"
                    f"  background-color: rgba({r},{g_c},{b},110);"
                    f"}}"
                    f"QPushButton:disabled {{ opacity: 0.3; }}"
                )
                btn.clicked.connect(
                    lambda checked, v=int(lbl): self.apply_label_requested.emit(v)
                )
                self._label_btns[int(lbl)] = btn
                row.addWidget(btn)
            l.addLayout(row)

        # Clear / Undo
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        self._clear_btn = QPushButton("Clear Selection")
        self._undo_btn  = QPushButton("Undo  (Ctrl+Z)")
        for b in (self._clear_btn, self._undo_btn):
            b.setEnabled(False)
            b.setFixedHeight(28)
            ctrl_row.addWidget(b)
        self._clear_btn.clicked.connect(self.clear_selection_requested)
        self._undo_btn.clicked.connect(self.undo_requested)
        l.addLayout(ctrl_row)

        return g

    def _build_legend_group(self) -> QGroupBox:
        g = QGroupBox("Legend")
        l = QVBoxLayout(g)
        l.setSpacing(3)
        for label in Label:
            row = QHBoxLayout()
            row.setSpacing(6)
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

    def _on_edit_toggle(self, checked: bool):
        self._edit_active = checked
        self._edit_toggle.setText(
            "Exit Edit Mode" if checked else "Enter Edit Mode"
        )
        self._edit_hint.setVisible(checked)
        self._sel_count.setVisible(checked)
        for btn in self._label_btns.values():
            btn.setEnabled(checked)
        self._clear_btn.setEnabled(checked)
        self.edit_mode_toggled.emit(checked)
        if not checked:
            self._sel_count.setText("No points selected")

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
        self._edit_toggle.setEnabled(False)
        if active and self._edit_active:
            self._edit_toggle.setChecked(False)

    def set_result(self, result) -> None:
        cyl    = result.cylinder
        counts = result.point_counts()

        def _f(v, unit="m", dp=2):
            return f"{v:.{dp}f} {unit}" if v is not None else "—"

        self._result_labels["Radius"].setText(_f(cyl.radius_m  if cyl else None))
        self._result_labels["Diameter"].setText(_f(cyl.diameter_m if cyl else None))
        self._result_labels["Height"].setText(_f(cyl.height_m  if cyl else None))
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
        self._edit_toggle.setEnabled(True)

    def update_selection_count(self, count: int) -> None:
        if count == 0:
            self._sel_count.setText("No points selected")
            for btn in self._label_btns.values():
                btn.setEnabled(self._edit_active)
        else:
            self._sel_count.setText(f"{count:,} points selected")
        self._undo_btn.setEnabled(True)

    def set_undo_available(self, available: bool) -> None:
        self._undo_btn.setEnabled(available)


# ── Helpers ────────────────────────────────────────────────────────────────

def _spin(lo, hi, step, val, suffix) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    return s


def _dim_label() -> QLabel:
    l = QLabel("—")
    l.setStyleSheet("color:#999;")
    return l
