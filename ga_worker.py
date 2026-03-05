from PyQt6.QtCore import QThread, pyqtSignal
from ga_engine import GAEngine
from ga_constants import GAConstants


class GAWorker(QThread):
    """
    QThread that runs the GA engine in background.
    Emits log messages and a done signal when optimization completes.
    """
    log_signal      = pyqtSignal(str)         # log message per epoch
    progress_signal = pyqtSignal(int)         # current epoch number
    done_signal     = pyqtSignal(float, int)  # (best_rms, n_active_images)

    def __init__(self, obj_points, matched_points, img_width, img_height, out_dir):
        super().__init__()
        self.obj_points     = obj_points
        self.matched_points = matched_points
        self.img_width      = img_width
        self.img_height     = img_height
        self.out_dir        = out_dir
        self._engine        = None

    def run(self):
        def on_progress(epoch, best_rms, msg):
            self.log_signal.emit(msg)
            if epoch is not None:
                self.progress_signal.emit(epoch)

        self._engine = GAEngine(
            obj_points=self.obj_points,
            matched_points=self.matched_points,
            img_width=self.img_width,
            img_height=self.img_height,
            out_dir=self.out_dir,
            progress_callback=on_progress
        )
        best_chromo, best_fit, K, dist = self._engine.run()
        self.done_signal.emit(best_fit, best_chromo.genes.count(True))

    def stop(self):
        if self._engine:
            self._engine.stop()
        self.wait()
