"""Left-hand control and results panel."""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QDoubleSpinBox, QFormLayout, QFrame,
    QSizePolicy, QSpacerItem,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from models.tank_scan import Label, LABEL_COLORS, LABEL_DISPLAY_NAMES


class ColourSwatch(QFrame):
    def __init__(self, rgb: tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        r, g, b = rgb
        self.setStyleSheet(f"background:{QColor(r,g,b).name()};border-radius:3px;")


class Sidebar(QWidget):
    open_requested    = Signal()
    segment_requested = Signal(str, dict)   # scan_type, params
    export_ply_requested  = Signal()
    export_json_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._result = None
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(12, 12, 12, 12)

        # ── File ──
        file_group = QGroupBox("Scan File")
        fl = QVBoxLayout(file_group)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color:#888;font-size:11px;")
        fl.addWidget(self._file_label)
        self._open_btn = QPushButton("Open E57…")
        self._open_btn.clicked.connect(self.open_requested)
        fl.addWidget(self._open_btn)
        root.addWidget(file_group)

        # ── Scan type ──
        type_group = QGroupBox("Scan Type")
        tl = QVBoxLayout(type_group)
        self._scan_combo = QComboBox()
        self._scan_combo.addItems(["External", "Internal"])
        tl.addWidget(self._scan_combo)
        type_desc = QLabel(
            "External: shell geometry, out-of-roundness, roof profile.\n"
            "Internal: floor plate, deadwood, heating coils."
        )
        type_desc.setWordWrap(True)
        type_desc.setStyleSheet("color:#888;font-size:10px;")
        tl.addWidget(type_desc)
        root.addWidget(type_group)

        # ── Processing options ──
        opts_group = QGroupBox("Processing Options")
        ol = QFormLayout(opts_group)
        ol.setSpacing(6)

        self._voxel_spin = QDoubleSpinBox()
        self._voxel_spin.setRange(0.005, 0.1)
        self._voxel_spin.setSingleStep(0.005)
        self._voxel_spin.setValue(0.01)
        self._voxel_spin.setSuffix(" m")
        ol.addRow("Voxel size:", self._voxel_spin)

        self._plane_spin = QDoubleSpinBox()
        self._plane_spin.setRange(0.005, 0.05)
        self._plane_spin.setSingleStep(0.005)
        self._plane_spin.setValue(0.015)
        self._plane_spin.setSuffix(" m")
        ol.addRow("Plane tolerance:", self._plane_spin)

        self._cyl_spin = QDoubleSpinBox()
        self._cyl_spin.setRange(0.005, 0.10)
        self._cyl_spin.setSingleStep(0.005)
        self._cyl_spin.setValue(0.02)
        self._cyl_spin.setSuffix(" m")
        ol.addRow("Cylinder tolerance:", self._cyl_spin)

        root.addWidget(opts_group)

        # ── Segment button ──
        self._segment_btn = QPushButton("Segment")
        self._segment_btn.setEnabled(False)
        self._segment_btn.setFixedHeight(36)
        self._segment_btn.clicked.connect(self._on_segment)
        root.addWidget(self._segment_btn)

        # ── Results ──
        self._results_group = QGroupBox("Results")
        rl = QFormLayout(self._results_group)
        rl.setSpacing(4)
        self._result_labels: dict[str, QLabel] = {}
        for key in ["Radius", "Diameter", "Height", "Floor Z", "Roof Z", "Total points"]:
            lbl = QLabel("—")
            lbl.setStyleSheet("color:#aaa;")
            rl.addRow(f"{key}:", lbl)
            self._result_labels[key] = lbl

        for label in Label:
            lbl = QLabel("—")
            lbl.setStyleSheet("color:#aaa;")
            rl.addRow(f"  {LABEL_DISPLAY_NAMES[label]}:", lbl)
            self._result_labels[label.name] = lbl

        root.addWidget(self._results_group)

        # ── Legend ──
        legend_group = QGroupBox("Legend")
        ll = QVBoxLayout(legend_group)
        ll.setSpacing(4)
        for label in Label:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(ColourSwatch(LABEL_COLORS[label]))
            row.addWidget(QLabel(LABEL_DISPLAY_NAMES[label]))
            row.addStretch()
            ll.addLayout(row)
        root.addWidget(legend_group)

        # ── Export ──
        export_group = QGroupBox("Export")
        el = QVBoxLayout(export_group)
        self._export_ply_btn  = QPushButton("Export Labeled PLY…")
        self._export_json_btn = QPushButton("Export Summary JSON…")
        for btn in (self._export_ply_btn, self._export_json_btn):
            btn.setEnabled(False)
            el.addWidget(btn)
        self._export_ply_btn.clicked.connect(self.export_ply_requested)
        self._export_json_btn.clicked.connect(self.export_json_requested)
        root.addWidget(export_group)

        root.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

    # ── Slots ─────────────────────────────────────────────────────────────
    def _on_segment(self):
        scan_type = self._scan_combo.currentText().lower()
        params = {
            "voxel_size": self._voxel_spin.value(),
            "plane_dist": self._plane_spin.value(),
            "cyl_tol":    self._cyl_spin.value(),
        }
        self.segment_requested.emit(scan_type, params)

    def set_file(self, path: str):
        import os
        self._file_label.setText(os.path.basename(path))
        self._segment_btn.setEnabled(True)

    def set_processing(self, active: bool):
        self._segment_btn.setEnabled(not active)
        self._open_btn.setEnabled(not active)
        self._export_ply_btn.setEnabled(False)
        self._export_json_btn.setEnabled(False)

    def set_result(self, result):
        self._result = result
        cyl = result.cylinder
        counts = result.point_counts()

        def _fmt(v, unit="m", decimals=2):
            return f"{v:.{decimals}f} {unit}" if v is not None else "—"

        self._result_labels["Radius"].setText(_fmt(cyl.radius_m if cyl else None))
        self._result_labels["Diameter"].setText(_fmt(cyl.diameter_m if cyl else None))
        self._result_labels["Height"].setText(_fmt(cyl.height_m if cyl else None))
        self._result_labels["Floor Z"].setText(_fmt(result.floor_z_m))
        self._result_labels["Roof Z"].setText(_fmt(result.roof_z_m))
        self._result_labels["Total points"].setText(f"{len(result.points):,}")

        for label in Label:
            n = counts.get(label.name, 0)
            pct = 100 * n / max(len(result.points), 1)
            self._result_labels[label.name].setText(f"{n:,}  ({pct:.1f}%)")

        self._segment_btn.setEnabled(True)
        self._export_ply_btn.setEnabled(True)
        self._export_json_btn.setEnabled(True)
