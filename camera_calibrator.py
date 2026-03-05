import cv2
import numpy as np
import os

class CameraCalibrator:
    def __init__(self):
        self.obj_points = []
        self.matched_points = []
        self.K = None
        self.dist_coeff = None
        self.new_camera_matrix = None
        self.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def add_points(self, pattern_points, matched_features):
        """Add points for future calibration."""
        self.obj_points.append(pattern_points)
        self.matched_points.append(matched_features)
        print(f"Points added. Total frames collected: {len(self.obj_points)}")

    def can_calibrate(self):
        return len(self.obj_points) > 0

    def calibrate(self, img_width, img_height):
        """Compute the calibration and return the RMS."""
        if not self.can_calibrate():
            print("No points for calibration!")
            return False

        rms, self.K, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.matched_points,
            (img_width, img_height),
            None,
            None
        )
        
        # Compute the optimal camera matrix
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist_coeff, (img_width, img_height), 1, (img_width, img_height)
        )
        
        self._save_to_xml("camera_calibration.xml", rvecs, tvecs)
        return rms

    def _save_to_xml(self, filename, rvecs, tvecs):
        """Function to save calibration data in XML format."""
        filepath = os.path.join(self.out_dir, filename)
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("cameraMatrix", self.K)
        fs.write("distCoeffs", self.dist_coeff)
        fs.write("rvecs", np.array(rvecs)) 
        fs.write("tvecs", np.array(tvecs))
        if self.new_camera_matrix is not None:
            fs.write("newCameraMatrix", self.new_camera_matrix)
        fs.release()
        print(f"Calibration data saved to {filename}")

    def compute_reprojection_errors(self, rvecs, tvecs):
        """Function to compute reprojection errors."""
        total_points = 0
        total_err = 0
        for i in range(len(self.obj_points)):
            img_pts2, _ = cv2.projectPoints(
                np.array(self.obj_points[i]), rvecs[i], tvecs[i], self.K, self.dist_coeff
            )
            img_pts2 = img_pts2.reshape(-1, 2)
            err = cv2.norm(np.array(self.matched_points[i]), img_pts2, cv2.NORM_L2)
            n = len(self.obj_points[i])
            total_err += err * err
            total_points += n
        return np.sqrt(total_err / total_points)
