import cv2
import numpy as np

class ChessboardValidator:
    def __init__(self, board_size=(9, 6), square_size=10):
        self.board_size = board_size
        self.square_size = square_size
        self.total_mean = 0.0
        self.tcount = 0
        
        # Pre-compute the board
        self.board = []
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                self.board.append((j * square_size, i * square_size))
        self.board = np.array(self.board, dtype=np.float32)

    def check_accuracy(self, image):
        """Finds the chessboard and prints the mean errors."""
        patternfound, corners = cv2.findChessboardCorners(
            image, self.board_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )
        if not patternfound:
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1), 
            (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.1)
        )

        H, _ = cv2.findHomography(self.board.reshape(-1,1,2), corners, cv2.RANSAC)
        if H is None:
            return

        proj = cv2.perspectiveTransform(self.board.reshape(-1,1,2), H).reshape(-1,2)
        sum_err = sum(np.linalg.norm(proj[i] - corners[i][0]) for i in range(len(proj)))

        mean_err = sum_err / len(proj)
        self.total_mean += mean_err
        self.tcount += 1
        print(f"Mean error: {mean_err:.4f} | Total mean stored: {self.total_mean / self.tcount:.4f}")
