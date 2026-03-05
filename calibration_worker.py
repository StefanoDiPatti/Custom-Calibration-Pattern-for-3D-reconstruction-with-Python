import cv2
from cv2 import ccalib
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from pattern_manager import PatternManager
from camera_calibrator import CameraCalibrator
from utils import ChessboardValidator
from ga_worker import GAWorker


def cv2_to_qpixmap(frame):
    """Convert an OpenCV BGR frame to QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Shared base widget: video label + log
# ---------------------------------------------------------------------------
class BaseCalibrationWidget(QWidget):
    def __init__(self, title, description, out_dir="out"):
        super().__init__()
        self.out_dir    = out_dir
        self._ga_worker = None  # GA QThread, kept as instance to avoid GC
        layout = QVBoxLayout(self)

        desc = QLabel(f"<b>{title}</b><br><small>{description}</small>")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        content = QHBoxLayout()
        layout.addLayout(content)

        # Video preview
        self.video_label = QLabel("No active feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background: #1a1a1a; color: white; border-radius: 6px;")
        content.addWidget(self.video_label, 3)

        # Right panel
        right_panel = QVBoxLayout()
        content.addLayout(right_panel, 1)

        # Calibration log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        log_layout.addWidget(self.log)
        right_panel.addWidget(log_group)

        # GA optimizer log (shown automatically during GA run)
        ga_log_group = QGroupBox("🧬 GA Optimizer Log")
        ga_log_layout = QVBoxLayout(ga_log_group)
        self.ga_log = QTextEdit()
        self.ga_log.setReadOnly(True)
        self.ga_log.setMaximumHeight(150)
        ga_log_layout.addWidget(self.ga_log)
        right_panel.addWidget(ga_log_group)

        # Status label
        self.status_label = QLabel("Status: Awaiting")
        self.status_label.setWordWrap(True)
        right_panel.addWidget(self.status_label)
        right_panel.addStretch()

        # Buttons populated by subclasses
        self.controls = QVBoxLayout()
        right_panel.addLayout(self.controls)

    def log_message(self, msg):
        self.log.append(msg)

    def update_frame(self, pixmap):
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _launch_ga(self, obj_points, matched_points, img_w, img_h):
        """Automatically launched after calibration completes in any mode."""
        self.ga_log.clear()
        self.ga_log.append("🧬 Starting GA optimization in background...")
        self.status_label.setText("Status: GA Optimization running...")

        self._ga_worker = GAWorker(obj_points, matched_points, img_w, img_h, self.out_dir)
        self._ga_worker.log_signal.connect(self.ga_log.append)
        self._ga_worker.done_signal.connect(self._on_ga_done)
        self._ga_worker.start()

    def _on_ga_done(self, best_rms, n_active):
        self.ga_log.append(
            f"\n✅ GA optimization completed!\n"
            f"   Best reprojection error : {best_rms:.5f}\n"
            f"   Optimal images selected : {n_active}\n"
            f"   Saved to                : {self.out_dir}/ga_best_calibration.xml"
        )
        self.status_label.setText(
            f"Status: GA done — Best RMS: {best_rms:.5f} | Images: {n_active}"
        )


# ---------------------------------------------------------------------------
# MODE 1: Real-Time Calibration
# ---------------------------------------------------------------------------
class RealtimeWorker(QThread):
    frame_ready      = pyqtSignal(QPixmap)
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    calibration_done = pyqtSignal(float)
    # Emitted after successful calibration to trigger GA automatically
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self):
        super().__init__()
        self.p_manager     = PatternManager()
        self.calibrator    = CameraCalibrator()
        self.validator     = ChessboardValidator()
        self._running      = False
        self._phase        = "select_roi"   # select_roi -> collect -> calibrated
        self._capture_next = False
        self._do_calibrate = False
        self._frame        = None

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._frame = frame.copy()

            if self._phase == "select_roi":
                canvas = self.p_manager.draw_roi(frame.copy())
                self.frame_ready.emit(cv2_to_qpixmap(canvas))

            elif self._phase == "collect":
                res, matched_feat, pattern_pts, out, H, corners = self.p_manager.find_pattern(frame)
                display = out if res else frame
                if res and self._capture_next:
                    self.calibrator.add_points(pattern_pts, matched_feat)
                    count = len(self.calibrator.obj_points)
                    self.log_signal.emit(f"Frame {count} acquired.")
                    self.status_signal.emit(f"Frames collected: {count}")
                    self._capture_next = False
                self.frame_ready.emit(cv2_to_qpixmap(display))

                if self._do_calibrate:
                    self._do_calibrate = False
                    h, w = frame.shape[:2]
                    rms = self.calibrator.calibrate(w, h)
                    if rms:
                        self.calibration_done.emit(rms)
                        self._phase = "calibrated"
                        # GA triggered automatically with all collected points
                        self.ga_ready.emit(
                            self.calibrator.obj_points,
                            self.calibrator.matched_points,
                            w, h
                        )

            elif self._phase == "calibrated":
                res, matched_feat, pattern_pts, out, H, corners = self.p_manager.find_pattern(frame)
                if res:
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.new_camera_matrix,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        frame = self.p_manager.pattern.drawOrientation(
                            frame, tvec, rvec,
                            self.calibrator.new_camera_matrix,
                            self.calibrator.dist_coeff, 50, 5
                        )
                self.frame_ready.emit(cv2_to_qpixmap(frame))

        cap.release()

    def confirm_roi(self):
        success, out = self.p_manager.create_pattern(self._frame)
        if success:
            self._phase = "collect"
            self.log_signal.emit("Pattern created. Click 'Capture Frame' to collect data.")
            self.status_signal.emit("Data collection mode active.")
        else:
            self.log_signal.emit("Error: Invalid ROI.")

    def capture_frame(self):
        self._capture_next = True

    def trigger_calibrate(self):
        self._do_calibrate = True

    def stop(self):
        self._running = False
        self.wait()


class RealtimeCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Real-Time Calibration",
            "Draw the ROI on the pattern with the mouse, then capture frames to calibrate."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker = None

        self.btn_start       = QPushButton("▶ Start Camera")
        self.btn_confirm_roi = QPushButton("✅ Confirm ROI")
        self.btn_capture     = QPushButton("📸 Capture Frame")
        self.btn_calibrate   = QPushButton("⚙️ Perform Calibration + GA")
        self.btn_stop        = QPushButton("⏹ Stop")

        self.btn_confirm_roi.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_calibrate.setEnabled(False)

        for btn in [self.btn_start, self.btn_confirm_roi, self.btn_capture,
                    self.btn_calibrate, self.btn_stop]:
            btn.setMinimumHeight(36)
            self.controls.addWidget(btn)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_confirm_roi.clicked.connect(self.confirm_roi)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_stop.clicked.connect(self.stop_worker)

        # Mouse tracking on video_label for ROI drawing
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent   = self._mouse_press
        self.video_label.mouseMoveEvent    = self._mouse_move
        self.video_label.mouseReleaseEvent = self._mouse_release

    def start_worker(self):
        self.worker = RealtimeWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(lambda m: self.status_label.setText(f"Status: {m}"))
        self.worker.calibration_done.connect(
            lambda rms: self.log_message(
                f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
            )
        )
        # GA starts automatically when the worker emits ga_ready
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_confirm_roi.setEnabled(True)
        self.log_message("Camera started. Draw the ROI on the pattern.")
        self.status_label.setText("Status: ROI Selection")

    def _scale_coords(self, x, y):
        """Convert QLabel pixel coordinates to real frame coordinates."""
        lw, lh = self.video_label.width(), self.video_label.height()
        fw, fh = 640, 480
        scale  = min(lw / fw, lh / fh)
        ox     = (lw - fw * scale) / 2
        oy     = (lh - fh * scale) / 2
        return max(0, int((x - ox) / scale)), max(0, int((y - oy) / scale))

    def _mouse_press(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, None, None)

    def _mouse_move(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_MOUSEMOVE, x, y, None, None)

    def _mouse_release(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_LBUTTONUP, x, y, None, None)

    def confirm_roi(self):
        if self.worker:
            self.worker.confirm_roi()
            self.btn_confirm_roi.setEnabled(False)
            self.btn_capture.setEnabled(True)
            self.btn_calibrate.setEnabled(True)

    def capture(self):
        if self.worker:
            self.worker.capture_frame()

    def calibrate(self):
        if self.worker:
            self.worker.trigger_calibrate()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.btn_start.setEnabled(True)


# ---------------------------------------------------------------------------
# MODE 2: Pattern Offline + Real-Time Photos
# ---------------------------------------------------------------------------
class OfflinePatternWorker(QThread):
    frame_ready      = pyqtSignal(QPixmap)
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    calibration_done = pyqtSignal(float)
    # Emitted after successful calibration to trigger GA automatically
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self, pattern_path):
        super().__init__()
        self.pattern_path  = pattern_path
        self.p_manager     = PatternManager()
        self.calibrator    = CameraCalibrator()
        self._running      = False
        self._phase        = "collect"
        self._capture_next = False
        self._do_calibrate = False
        self._frame        = None
        self.img_w         = 640
        self.img_h         = 480

    def run(self):
        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            self.log_signal.emit("Error: Unable to load the pattern.")
            return

        out = np.zeros_like(pattern_img)
        if not self.p_manager.pattern.create(pattern_img, pattern_img.shape[:2], out):
            self.log_signal.emit("Error: Pattern creation failed.")
            return

        self.log_signal.emit("Pattern loaded successfully.")
        self._running = True
        cap = cv2.VideoCapture(0)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._frame = frame.copy()
            self.img_h, self.img_w = frame.shape[:2]

            if self._phase == "collect":
                res, matched_feat, pattern_pts, out_frame, H, corners = self.p_manager.find_pattern(frame)
                display = out_frame if res else frame
                if res and self._capture_next:
                    self.calibrator.add_points(pattern_pts, matched_feat)
                    count = len(self.calibrator.obj_points)
                    self.log_signal.emit(f"Frame {count} acquired.")
                    self.status_signal.emit(f"Frames collected: {count}")
                    self._capture_next = False
                self.frame_ready.emit(cv2_to_qpixmap(display))

                if self._do_calibrate:
                    self._do_calibrate = False
                    rms = self.calibrator.calibrate(self.img_w, self.img_h)
                    if rms:
                        self.calibration_done.emit(rms)
                        self._phase = "calibrated"
                        # GA triggered automatically with all collected points
                        self.ga_ready.emit(
                            self.calibrator.obj_points,
                            self.calibrator.matched_points,
                            self.img_w, self.img_h
                        )

            elif self._phase == "calibrated":
                res, matched_feat, pattern_pts, out_frame, H, corners = self.p_manager.find_pattern(frame)
                if res:
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.new_camera_matrix,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        frame = self.p_manager.pattern.drawOrientation(
                            frame, tvec, rvec,
                            self.calibrator.new_camera_matrix,
                            self.calibrator.dist_coeff, 50, 5
                        )
                self.frame_ready.emit(cv2_to_qpixmap(frame))

        cap.release()

    def capture_frame(self):
        self._capture_next = True

    def trigger_calibrate(self):
        self._do_calibrate = True

    def stop(self):
        self._running = False
        self.wait()


class OfflinePatternCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Pattern Offline + Real-Time Photos",
            "Load a pattern image from file, then take photos with the camera to calibrate."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker       = None
        self.pattern_path = None

        self.btn_load_pattern = QPushButton("📂 Load Pattern")
        self.pattern_info     = QLabel("No pattern loaded")
        self.pattern_info.setStyleSheet("color: gray; font-style: italic;")
        self.btn_start     = QPushButton("▶ Start Camera")
        self.btn_capture   = QPushButton("📸 Capture Frame")
        self.btn_calibrate = QPushButton("⚙️ Perform Calibration + GA")
        self.btn_stop      = QPushButton("⏹ Stop")

        self.btn_start.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_calibrate.setEnabled(False)

        for w in [self.btn_load_pattern, self.pattern_info, self.btn_start,
                  self.btn_capture, self.btn_calibrate, self.btn_stop]:
            if isinstance(w, QPushButton):
                w.setMinimumHeight(36)
            self.controls.addWidget(w)

        self.btn_load_pattern.clicked.connect(self.load_pattern)
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_stop.clicked.connect(self.stop_worker)

    def load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pattern Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.pattern_path = path
            self.pattern_info.setText(f"Pattern: {path.split('/')[-1]}")
            self.pattern_info.setStyleSheet("color: green;")
            self.btn_start.setEnabled(True)
            self.log_message(f"Pattern selected: {path}")

    def start_worker(self):
        self.worker = OfflinePatternWorker(self.pattern_path)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(lambda m: self.status_label.setText(f"Status: {m}"))
        self.worker.calibration_done.connect(
            lambda rms: self.log_message(
                f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
            )
        )
        # GA starts automatically when the worker emits ga_ready
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_calibrate.setEnabled(True)
        self.status_label.setText("Status: Collecting frames")

    def capture(self):
        if self.worker:
            self.worker.capture_frame()

    def calibrate(self):
        if self.worker:
            self.worker.trigger_calibrate()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.btn_start.setEnabled(True)


# ---------------------------------------------------------------------------
# MODE 3: Offline Calibration
# ---------------------------------------------------------------------------
class FullOfflineWorker(QThread):
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    result_frame     = pyqtSignal(QPixmap)
    calibration_done = pyqtSignal(float)
    # Emitted after successful calibration to trigger GA automatically
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self, pattern_path, image_paths):
        super().__init__()
        self.pattern_path = pattern_path
        self.image_paths  = image_paths
        self.p_manager    = PatternManager()
        self.calibrator   = CameraCalibrator()
        self.img_w = 0
        self.img_h = 0

    def run(self):
        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            self.log_signal.emit("Error: Unable to load the pattern.")
            return

        out = np.zeros_like(pattern_img)
        if not self.p_manager.pattern.create(pattern_img, pattern_img.shape[:2], out):
            self.log_signal.emit("Error: Failed to create the pattern.")
            return
        self.log_signal.emit("Pattern loaded successfully.")

        for i, path in enumerate(self.image_paths):
            img = cv2.imread(path)
            if img is None:
                self.log_signal.emit(f"[SKIP] Error reading: {path}")
                continue

            self.img_h, self.img_w = img.shape[:2]
            res, matched_feat, pattern_pts, out_frame, H, corners = self.p_manager.find_pattern(img)

            if res and len(matched_feat) > 3:
                self.calibrator.add_points(pattern_pts, matched_feat)
                self.log_signal.emit(f"[{i+1}/{len(self.image_paths)}] Pattern found ✅")
                self.result_frame.emit(cv2_to_qpixmap(out_frame))
            else:
                self.log_signal.emit(f"[{i+1}/{len(self.image_paths)}] Pattern not found ❌")

        n_valid = len(self.calibrator.obj_points)
        self.status_signal.emit(f"Analysis completed. Valid frames: {n_valid}")

        if self.calibrator.can_calibrate():
            rms = self.calibrator.calibrate(self.img_w, self.img_h)
            if rms:
                self.calibration_done.emit(rms)
                # GA triggered automatically with all collected points
                self.ga_ready.emit(
                    self.calibrator.obj_points,
                    self.calibrator.matched_points,
                    self.img_w, self.img_h
                )
        else:
            self.log_signal.emit("No valid frames for calibration.")


class FullOfflineCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Offline Calibration",
            "Load the pattern and a set of calibration images from disk. No camera required."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker       = None
        self.pattern_path = None
        self.image_paths  = []

        self.btn_load_pattern = QPushButton("📂 Load Pattern")
        self.pattern_info     = QLabel("No pattern loaded")
        self.pattern_info.setStyleSheet("color: gray; font-style: italic;")

        self.btn_load_images = QPushButton("🗂️ Load Calibration Images")
        self.images_info     = QLabel("No images loaded")
        self.images_info.setStyleSheet("color: gray; font-style: italic;")

        self.btn_run = QPushButton("⚙️ Run Calibration + GA")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")

        for w in [self.btn_load_pattern, self.pattern_info,
                  self.btn_load_images, self.images_info,
                  self.btn_run]:
            if isinstance(w, QPushButton):
                w.setMinimumHeight(36)
            self.controls.addWidget(w)

        self.btn_load_pattern.clicked.connect(self.load_pattern)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_run.clicked.connect(self.run_calibration)

    def load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pattern image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.pattern_path = path
            self.pattern_info.setText(f"Pattern: {path.split('/')[-1]}")
            self.pattern_info.setStyleSheet("color: green;")
            self.log_message(f"Pattern: {path}")
            self._check_ready()

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select calibration images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if paths:
            self.image_paths = paths
            self.images_info.setText(f"{len(paths)} images loaded")
            self.images_info.setStyleSheet("color: green;")
            self.log_message(f"Loaded {len(paths)} images.")
            self._check_ready()

    def _check_ready(self):
        if self.pattern_path and self.image_paths:
            self.btn_run.setEnabled(True)

    def run_calibration(self):
        self.btn_run.setEnabled(False)
        self.worker = FullOfflineWorker(self.pattern_path, self.image_paths)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(lambda m: self.status_label.setText(f"Status: {m}"))
        self.worker.result_frame.connect(self.update_frame)
        self.worker.calibration_done.connect(
            lambda rms: (
                self.log_message(f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."),
                self.btn_run.setEnabled(True)
            )
        )
        # GA starts automatically when the worker emits ga_ready
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.status_label.setText("Status: Processing...")
