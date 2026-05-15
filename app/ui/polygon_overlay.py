"""
Transparent overlay widget drawn on top of the VTK render window.
Shows the in-progress polygon lasso (vertices + edges + rubber line to cursor).
All mouse events pass through unchanged.
"""
from __future__ import annotations
from typing import Optional

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PySide6.QtCore import QPointF


_VERTEX_COLOUR  = QColor(255, 210, 40, 230)
_EDGE_COLOUR    = QColor(255, 210, 40, 200)
_CLOSE_COLOUR   = QColor(255, 210, 40, 100)
_FILL_COLOUR    = QColor(255, 210, 40,  25)
_VERTEX_RADIUS  = 5


class PolygonOverlay(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setVisible(False)

        self._vertices: list[tuple[int, int]] = []
        self._cursor:   Optional[tuple[int, int]] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def set_vertices(
        self,
        vertices: list[tuple[int, int]],
        cursor: Optional[tuple[int, int]] = None,
    ) -> None:
        self._vertices = vertices
        self._cursor   = cursor
        self.setVisible(bool(vertices))
        self.update()

    def clear(self) -> None:
        self._vertices = []
        self._cursor   = None
        self.setVisible(False)
        self.update()

    # ── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        if not self._vertices:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        verts = [QPointF(x, y) for x, y in self._vertices]

        # Filled polygon preview (if ≥ 3 vertices)
        if len(verts) >= 3:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(_FILL_COLOUR))
            painter.drawPolygon(QPolygonF(verts))

        # Drawn edges
        painter.setPen(QPen(_EDGE_COLOUR, 1.8))
        painter.setBrush(Qt.NoBrush)
        for i in range(len(verts) - 1):
            painter.drawLine(verts[i], verts[i + 1])

        # Rubber line to cursor
        if self._cursor and verts:
            painter.setPen(QPen(_CLOSE_COLOUR, 1.4, Qt.DashLine))
            painter.drawLine(verts[-1], QPointF(*self._cursor))
            # Close-loop preview line
            if len(verts) >= 2:
                painter.setPen(QPen(_CLOSE_COLOUR, 1.0, Qt.DotLine))
                painter.drawLine(QPointF(*self._cursor), verts[0])

        # Vertex dots
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_VERTEX_COLOUR))
        for v in verts:
            painter.drawEllipse(
                v.x() - _VERTEX_RADIUS, v.y() - _VERTEX_RADIUS,
                _VERTEX_RADIUS * 2,     _VERTEX_RADIUS * 2,
            )
        # First vertex is larger (close target)
        if len(verts) >= 3:
            painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
            r = _VERTEX_RADIUS + 3
            painter.drawEllipse(verts[0].x() - r, verts[0].y() - r, r * 2, r * 2)

        painter.end()
