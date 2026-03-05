import cv2
import numpy as np

class PatternManager:
    def __init__(self):
        self.pattern = cv2.ccalib.CustomPattern()
        self.roi = [0, 0, 0, 0]  # [x, y, width, height]
        self.mdown = False

    def on_mouse(self, event, x, y, flags, param):
        """Callback for mouse selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi[0], self.roi[1] = x, y
            self.roi[2], self.roi[3] = 0, 0
            self.mdown = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi[2] = x - self.roi[0]
            self.roi[3] = y - self.roi[1]
            print("ROI Selected:", self.roi)
            self.mdown = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mdown:
            self.roi[2] = x - self.roi[0]
            self.roi[3] = y - self.roi[1]

    def draw_roi(self, frame):
        """Draws the rectangle on the frame during selection."""
        x, y, w, h = self.roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def create_pattern(self, frame):
        """Cut the ROI and create the pattern."""
        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            return False, None
            
        pattern_roi = frame[y:y+h, x:x+w]
        out = np.zeros_like(pattern_roi)
        success = self.pattern.create(pattern_roi, pattern_roi.shape[:2], out)
        return success, out

    def find_pattern(self, frame, ratio=0.7, proj_error=8.0):
        """Finds the pattern in the provided frame."""
        return self.pattern.findPattern(
            frame, ratio=ratio, proj_error=proj_error, refine_position=False
        )
