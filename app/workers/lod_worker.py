"""Background QThread worker to build LOD pyramid after segmentation."""
from PySide6.QtCore import QObject, QThread, Signal, Slot
from processing.lod import PointCloudLOD
import numpy as np


class LODWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)   # PointCloudLOD
    error    = Signal(str)

    def __init__(self, pts: np.ndarray):
        super().__init__()
        self._pts = pts

    @Slot()
    def run(self) -> None:
        try:
            lod = PointCloudLOD(pts=self._pts)
            lod.build(progress_cb=lambda msg: self.progress.emit(msg))
            self.finished.emit(lod)
        except Exception as exc:
            self.error.emit(str(exc))
