"""Entry point for the API 653 Tank Inspector desktop application."""
import sys
import os

# Ensure the app/ directory is on the path so all packages resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Tank Inspector")
    app.setOrganizationName("API653")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
