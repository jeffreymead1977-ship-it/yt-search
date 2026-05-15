"""Background QThread worker to build octree LOD after segmentation."""
from PySide6.QtCore import QObject, Signal, Slot
from processing.octree_lod import build_octree, OctreeNode
import numpy as np


class LODWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)   # OctreeNode root
    error    = Signal(str)

    def __init__(self, pts: np.ndarray):
        super().__init__()
        self._pts = pts

    @Slot()
    def run(self) -> None:
        try:
            root = build_octree(
                self._pts,
                max_depth=8,
                min_pts=500,
                samples_per_node=150,
                progress_cb=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(root)
        except Exception as exc:
            self.error.emit(str(exc))
