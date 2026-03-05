import cv2
import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QIcon
from calibration_worker import (
    RealtimeCalibrationWidget,
    OfflinePatternCalibrationWidget,
    FullOfflineCalibrationWidget
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "res")
OUT_DIR  = os.path.join(BASE_DIR, "out")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Pattern Camera Calibration Tool")
        self.setMinimumSize(900, 700)
        self.setWindowIcon(QIcon(os.path.join(RES_DIR, "camera_calib_tool_icon.png")))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header with logo + title
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        pixmap = QPixmap(os.path.join(RES_DIR, "camera_calib_tool_icon.png"))
        pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        header_layout.addWidget(logo_label)

        title = QLabel("Camera Calibration Tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(title)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Mode selection buttons (GA is integrated into each mode automatically)
        btn_layout = QHBoxLayout()
        self.btn_realtime        = QPushButton("📷 Real-Time Calibration")
        self.btn_offline_pattern = QPushButton("🖼️ Pattern Offline + Real-Time Photos")
        self.btn_full_offline    = QPushButton("📁 Offline Calibration")

        for btn in [self.btn_realtime, self.btn_offline_pattern, self.btn_full_offline]:
            btn.setMinimumHeight(50)
            btn.setFont(QFont("Arial", 11))
            btn.setCheckable(True)
            btn_layout.addWidget(btn)

        main_layout.addLayout(btn_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(line)

        # Stack with the 3 widgets
        self.stack     = QStackedWidget()
        self.rt_widget = RealtimeCalibrationWidget(out_dir=OUT_DIR)
        self.op_widget = OfflinePatternCalibrationWidget(out_dir=OUT_DIR)
        self.fo_widget = FullOfflineCalibrationWidget(out_dir=OUT_DIR)

        self.stack.addWidget(self.rt_widget)   # index 0
        self.stack.addWidget(self.op_widget)   # index 1
        self.stack.addWidget(self.fo_widget)   # index 2
        main_layout.addWidget(self.stack)

        # Connections
        self.btn_realtime.clicked.connect(lambda: self._switch(0))
        self.btn_offline_pattern.clicked.connect(lambda: self._switch(1))
        self.btn_full_offline.clicked.connect(lambda: self._switch(2))

        self._switch(0)

    def _switch(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate([self.btn_realtime, self.btn_offline_pattern, self.btn_full_offline]):
            btn.setChecked(i == index)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
