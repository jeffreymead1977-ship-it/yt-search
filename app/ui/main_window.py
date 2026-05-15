"""Main application window."""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QStatusBar, QProgressBar, QFileDialog, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction

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
    background-color: #3d4265;
    color: #e0e0e0;
    border: 1px solid #4d5275;
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 24px;
}
QPushButton:hover    { background-color: #4d5285; }
QPushButton:pressed  { background-color: #2d3255; }
QPushButton:disabled { background-color: #23262f; color: #555; border-color: #2d3040; }
QPushButton#primary  { background-color: #5b6aff; border-color: #6b7aff; }
QPushButton#primary:hover   { background-color: #6b7aff; }
QPushButton#primary:pressed { background-color: #4b5aef; }
QComboBox {
    background: #23262f;
    border: 1px solid #3d4060;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #23262f; selection-background-color: #3d4265; }
QDoubleSpinBox {
    background: #23262f;
    border: 1px solid #3d4060;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 22px;
}
QSplitter::handle { background: #2d3040; width: 2px; }
QStatusBar { background: #13151f; border-top: 1px solid #2d3040; }
QProgressBar {
    border: 1px solid #2d3040;
    border-radius: 3px;
    background: #13151f;
    max-height: 6px;
    text-align: center;
}
QProgressBar::chunk { background: #5b6aff; border-radius: 3px; }
QScrollBar:vertical {
    background: #13151f; width: 8px; margin: 0;
}
QScrollBar::handle:vertical { background: #3d4060; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMenuBar { background: #13151f; border-bottom: 1px solid #2d3040; }
QMenuBar::item:selected { background: #2d3040; }
QMenu { background: #1a1d27; border: 1px solid #2d3040; }
QMenu::item:selected { background: #3d4265; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("API 653 Tank Inspector")
        self.resize(1280, 800)
        self.setStyleSheet(APP_STYLESHEET)

        self._current_file: str | None = None
        self._current_result = None
        self._thread: QThread | None = None
        self._worker: SegmentationWorker | None = None

        self._build_menu()
        self._build_central()
        self._build_status_bar()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        open_act = QAction("&Open E57…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)

        file_menu.addSeparator()
        export_ply  = QAction("Export Labeled &PLY…", self)
        export_json = QAction("Export Summary &JSON…", self)
        export_ply.triggered.connect(self._export_ply)
        export_json.triggered.connect(self._export_json)
        file_menu.addAction(export_ply)
        file_menu.addAction(export_json)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = menu.addMenu("&View")
        reset_cam = QAction("&Reset Camera", self)
        reset_cam.setShortcut("R")
        reset_cam.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_cam)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)

        self._sidebar = Sidebar()
        self._sidebar.open_requested.connect(self._open_file)
        self._sidebar.segment_requested.connect(self._start_segmentation)
        self._sidebar.export_ply_requested.connect(self._export_ply)
        self._sidebar.export_json_requested.connect(self._export_json)

        self._viewer = ViewerWidget()

        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._viewer)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_label = QLabel("Ready")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate
        self._progress_bar.setFixedWidth(160)
        self._progress_bar.setVisible(False)
        sb.addWidget(self._status_label)
        sb.addPermanentWidget(self._progress_bar)

    # ── Actions ───────────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open E57 Scan", "", "E57 Point Cloud (*.e57)"
        )
        if path:
            self._current_file = path
            self._current_result = None
            self._sidebar.set_file(path)
            self._status("File loaded — click Segment to process.")

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
            f"Done — {len(result.points):,} points  |  "
            f"Shell {counts['SHELL']:,}  Floor {counts['FLOOR']:,}  "
            f"Roof {counts['ROOF']:,}  Deadwood {counts['DEADWOOD']:,}  "
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

    def _export_ply(self):
        if not self._current_result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Labeled PLY", "tank_labeled.ply", "PLY Files (*.ply)"
        )
        if path:
            from processing.exporter import export_labeled_ply
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
