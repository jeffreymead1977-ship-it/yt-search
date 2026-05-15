"""QThread worker — keeps the UI responsive during heavy processing."""
from PySide6.QtCore import QObject, QThread, Signal, Slot
from models.tank_scan import SegmentationResult


class SegmentationWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)   # SegmentationResult
    error    = Signal(str)

    def __init__(self, e57_path: str, scan_type: str, params: dict):
        super().__init__()
        self._path      = e57_path
        self._scan_type = scan_type
        self._params    = params

    @Slot()
    def run(self) -> None:
        try:
            from processing.e57_loader import load_e57
            from processing.segmenter  import segment

            self.progress.emit("Loading E57 file…")
            xyz, metadata = load_e57(self._path)
            self.progress.emit(f"Loaded {len(xyz):,} points. Starting segmentation…")

            result = segment(
                xyz,
                scan_type=self._scan_type,
                progress_cb=lambda msg: self.progress.emit(msg),
                **self._params,
            )
            result.scan_metadata = metadata
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))
