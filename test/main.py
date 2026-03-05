import cv2
from pattern_manager import PatternManager
from camera_calibrator import CameraCalibrator
from utils import ChessboardValidator

def main():
    video = cv2.VideoCapture(0)
    
    # Initialize our classes
    p_manager = PatternManager()
    calibrator = CameraCalibrator()
    validator = ChessboardValidator()

    # --- Selection of ROI ---
    cv2.namedWindow("Select Pattern")
    cv2.setMouseCallback("Select Pattern", p_manager.on_mouse)

    print("Draw a rectangle and press 't' to finish selection.")
    while True:
        ret, frame = video.read()
        if not ret: break

        canvas = p_manager.draw_roi(frame.copy())
        cv2.imshow("Select Pattern", canvas)
        
        # Validator for the chessboard pattern - shows feedback on the selection
        validator.check_accuracy(frame)

        if cv2.waitKey(10) & 0xFF == ord('t'):
            break

    cv2.destroyWindow("Select Pattern")

    # --- Pattern Creation ---
    success, out_pattern = p_manager.create_pattern(frame)
    if not success:
        print("Error creating pattern. Did you select a valid ROI?")
        return

    cv2.imshow("Pattern Created", out_pattern)
    cv2.waitKey(0)

    # --- Collection of Points for Calibration ---
    print("Press 's' to save the pattern points, 'q' to stop collecting.")
    while True:
        ret, frame = video.read()
        if not ret: break

        # Search for the pattern using the p_manager
        res, matched_feat, pattern_pts, out, H, corners = p_manager.find_pattern(frame)

        if res:
            cv2.imshow("Pattern Found", out)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') and res and len(matched_feat) > 3:
            # Add points to calibrator
            calibrator.add_points(pattern_pts, matched_feat)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    # --- Execution of Calibration ---
    img_width, img_height = frame.shape[1], frame.shape[0]
    rms = calibrator.calibrate(img_width, img_height)
    if not rms:
        return
    print(f"Calibration completed with RMS: {rms}")

    # --- Tracking e Axis Drawing (Pose Estimation) ---
    print("Calibration finished. Showing 3D axes. Press 'q' to exit, 's' to save frame.")
    save_counter = 0
    while True:
        ret, frame = video.read()
        if not ret: break

        res, matched_feat, pattern_pts, out, H, corners = p_manager.find_pattern(frame)

        if res:
            # Compute R and t using parameters from calibration
            success, rvec, tvec = p_manager.pattern.findRt(
                pattern_pts, matched_feat, 
                calibrator.new_camera_matrix, 
                calibrator.dist_coeff, None, None
            )

            if success:
                output = p_manager.pattern.drawOrientation(
                    frame, tvec, rvec, calibrator.new_camera_matrix, calibrator.dist_coeff, 50, 5
                )
                cv2.imshow("3D Axes", output)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('s'):
                    cv2.imwrite(f"axis_{save_counter}.png", output)
                    print(f"Saved axis_{save_counter}.png")
                    save_counter += 1
                elif key == ord('q'):
                    break
        else:
            cv2.imshow("3D Axes", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
