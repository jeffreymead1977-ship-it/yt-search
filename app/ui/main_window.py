"""Main application window."""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QStatusBar,
    QProgressBar, QFileDialog, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QKeySequence, QShortcut

from ui.sidebar       import Sidebar
from ui.viewer_widget import ViewerWidget
from workers.segmentation_worker import SegmentationWorker


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1d27;
    color: #e0e0e0;
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2d3040;
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; }
QPushButton {
    background-color: #2e3245;
    color: #d0d0d0;
    border: 1px solid #3d4265;
    border-radius: 4px;
    padding: 5px 12px;
    min-height: 22px;
}
QPushButton:hover    { background-color: #3d4265; }
QPushButton:pressed  { background-color: #232840; }
QPushButton:disabled { background-color: #1e2030; color: #4a4a5a; border-color: #2a2d3e; }
QPushButton#primary  { background-color: #3d4ecc; border-color: #5060dd; color: #fff; }
QPushButton#primary:hover   { background-color: #4d5edc; }
QPushButton#primary:pressed { background-color: #2d3ebc; }
QPushButton:checked {
    background-color: #3a5a3a;
    border-color: #4a8a4a;
    color: #aaeeaa;
}
QComboBox {
    background: #1e2030;
    border: 1px solid #3d4060;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1e2030; selection-background-color: #3d4265; }
QDoubleSpinBox {
    background: #1e2030;
    border: 1px solid #3d4060;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 20px;
}
QSplitter::handle { background: #2d3040; width: 2px; }
QStatusBar { background: #13151f; border-top: 1px solid #2d3040; color: #aaa; font-size: 11px; }
QProgressBar {
    border: 1px solid #2d3040;
    border-radius: 3px;
    background: #13151f;
    max-height: 6px;
}
QProgressBar::chunk { background: #3d4ecc; border-radius: 3px; }
QScrollBar:vertical { background: #13151f; width: 7px; }
QScrollBar::handle:vertical { background: #3d4060; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMenuBar { background: #13151f; border-bottom: 1px solid #2d3040; }
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background: #2d3040; }
QMenu { background: #1a1d27; border: 1px solid #2d3040; }
QMenu::item { padding: 5px 20px; }
QMenu::item:selected { background: #2d3255; }
QMenu::separator { background: #2d3040; height: 1px; margin: 3px 0; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("API 653 Tank Inspector")
        self.resize(1300, 820)
        self.setStyleSheet(APP_STYLESHEET)

        self._current_file: str | None = None
        self._current_result = None
        self._thread: QThread | None = None
        self._worker: SegmentationWorker | None = None

        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self._build_shortcuts()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        self._add_action(file_menu, "&Open E57…",          self._open_file,    "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "Export Labeled &PLY…", self._export_ply)
        self._add_action(file_menu, "Export Summary &JSON…",self._export_json)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Quit",                self.close,         "Ctrl+Q")

        edit_menu = mb.addMenu("&Edit")
        self._add_action(edit_menu, "&Undo",                self._undo,         "Ctrl+Z")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "Clear &Selection",     self._clear_selection)

        view_menu = mb.addMenu("&View")
        self._add_action(view_menu, "&Reset Camera",        self._reset_camera, "R")

    def _add_action(self, menu, label, slot, shortcut=None):
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)

        self._sidebar = Sidebar()
        self._sidebar.open_requested.connect(self._open_file)
        self._sidebar.segment_requested.connect(self._start_segmentation)
        self._sidebar.tool_activated.connect(self._on_tool_activated)
        self._sidebar.apply_label_requested.connect(self._apply_label)
        self._sidebar.clear_selection_requested.connect(self._clear_selection)
        self._sidebar.reset_box_requested.connect(self._viewer.reset_limit_box)
        self._sidebar.undo_requested.connect(self._undo)
        self._sidebar.wand_radius_changed.connect(self._viewer.set_wand_radius)
        self._sidebar.wand_same_label_changed.connect(self._viewer.set_wand_same_label)
        self._sidebar.export_ply_requested.connect(self._export_ply)
        self._sidebar.export_json_requested.connect(self._export_json)

        self._viewer = ViewerWidget()
        self._viewer.selection_changed.connect(self._on_selection_changed)

        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("Ready")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFixedWidth(150)
        self._progress_bar.setVisible(False)
        sb.addWidget(self._status_label)
        sb.addPermanentWidget(self._progress_bar)

    def _build_shortcuts(self):
        # Ctrl+Z also available outside the menu (catches focus in viewer)
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self._undo)

    # ── File & segmentation ────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open E57 Scan", "", "E57 Point Cloud (*.e57)"
        )
        if path:
            self._current_file   = path
            self._current_result = None
            self._sidebar.set_file(path)
            self._status("File loaded — configure options and click Segment.")

    def _start_segmentation(self, scan_type: str, params: dict):
        if not self._current_file:
            return
        if self._thread and self._thread.isRunning():
            return

        self._sidebar.set_processing(True)
        self._progress_bar.setVisible(True)
        self._status("Processing…")

        self._thread = QThread(self)
        self._worker = SegmentationWorker(self._current_file, scan_type, params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._status)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_done(self, result):
        self._current_result = result
        self._sidebar.set_result(result)
        self._viewer.display_result(result)
        self._progress_bar.setVisible(False)
        counts = result.point_counts()
        self._status(
            f"Segmentation complete — {len(result.points):,} pts  |  "
            f"Shell {counts['SHELL']:,}  "
            f"Floor {counts['FLOOR']:,}  "
            f"Roof {counts['ROOF']:,}  "
            f"Deadwood {counts['DEADWOOD']:,}  "
            f"Rubbish {counts['RUBBISH']:,}"
        )

    def _on_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self._sidebar.set_processing(False)
        self._status(f"Error: {msg}")
        QMessageBox.critical(self, "Segmentation Error", msg)

    def _cleanup_thread(self):
        self._sidebar.set_processing(False)
        self._thread = None
        self._worker = None

    # ── Edit tools ─────────────────────────────────────────────────────────

    def _on_tool_activated(self, tool: str):
        self._viewer.set_tool(tool)
        hints = {
            "polygon":  "Polygon — click to add vertices, double-click or Enter to finish.  "
                        "Drag as normal to orbit.",
            "limit_box":"Limit Box — drag the handles in the 3D view to clip the cloud.",
            "wand":     "Magic Wand — click a point to grow a radius selection.",
            "none":     "Ready.",
        }
        self._status(hints.get(tool, "Ready."))

    def _on_selection_changed(self, count: int):
        self._sidebar.update_selection_count(count)
        self._sidebar.set_undo_available(self._viewer.can_undo())

    def _apply_label(self, label_value: int):
        self._viewer.apply_label(label_value)
        self._sidebar.set_undo_available(self._viewer.can_undo())
        from models.tank_scan import Label
        name = Label(label_value).name
        self._status(f"Relabelled selection → {name}")

    def _clear_selection(self):
        self._viewer.clear_selection()

    def _undo(self):
        self._viewer.undo()
        self._sidebar.set_undo_available(self._viewer.can_undo())
        self._status("Undo.")

    # ── Export & view ──────────────────────────────────────────────────────

    def _export_ply(self):
        if not self._current_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Labeled PLY", "tank_labeled.ply", "PLY Files (*.ply)"
        )
        if path:
            from processing.exporter import export_labeled_ply
            # If the user has manually edited labels in the viewer, pull them back
            if self._viewer._labels is not None:
                self._current_result.labels = self._viewer._labels.copy()
            export_labeled_ply(self._current_result, path)
            self._status(f"PLY exported → {path}")

    def _export_json(self):
        if not self._current_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary JSON", "tank_summary.json", "JSON Files (*.json)"
        )
        if path:
            from processing.exporter import export_summary
            if self._viewer._labels is not None:
                self._current_result.labels = self._viewer._labels.copy()
            export_summary(self._current_result, path)
            self._status(f"JSON exported → {path}")

    def _reset_camera(self):
        if self._viewer._plotter:
            self._viewer._plotter.reset_camera()

    def _status(self, msg: str):
        self._status_label.setText(msg)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
