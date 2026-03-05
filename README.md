# 📷 Custom Pattern Camera Calibration Tool

A PyQt6-based desktop application for monocular camera calibration using
custom (non-chessboard) patterns. The tool integrates a **Genetic Algorithm (GA)**
optimizer that automatically selects the optimal subset of calibration images
and OpenCV calibration flags, minimizing the reprojection error.

---

## ✨ Features

- **3 calibration modes** in a unified GUI:
  - 📷 **Real-Time**: draw ROI live on the camera feed, capture frames and calibrate
  - 🖼️ **Pattern Offline + Real-Time Photos**: load a pattern image from disk, capture frames with the camera
  - 📁 **Full Offline**: load pattern + a set of images entirely from disk, no camera needed
- **Automatic GA optimization** after every calibration: no manual step required
- Custom pattern support via OpenCV `ccalib` (`cv2.ccalib.CustomPattern`)
- Results saved as `.xml` files (OpenCV `FileStorage` format)
- Per-mode **GA log panel** embedded in the UI, showing live epoch progress

---

## 🧬 How the GA works

After the standard `cv2.calibrateCamera()` call completes, the app automatically
launches a `GAWorker` thread that runs a **Genetic Algorithm** on the full set of
collected calibration frames.

Each **chromosome** encodes:
- A binary mask over the collected frames (which images to include)
- A set of OpenCV calibration flags (`CALIB_FIX_K3`, `CALIB_RATIONAL_MODEL`, etc.)

The **fitness function** runs `cv2.calibrateCamera()` on the active subset and
computes the mean reprojection error. The GA evolves the population over multiple
epochs using crossover, mutation, and a cataclysm mechanism to preserve diversity.

The best result is saved to `out/ga_best_calibration.xml`, containing:
- `cameraMatrix`, `newCameraMatrix`, `distCoeffs`
- `ERR_REPROJ`, `N_IMAGES`, `ACTIVE_INDICES`, `FLAGS`

---

## 📁 Project Structure

```text
├── app.py                  # Main entry point, MainWindow (PyQt6)
├── calibration_worker.py   # All 3 calibration mode widgets + workers (QThread)
├── ga_worker.py            # GAWorker QThread (wraps GAEngine)
├── ga_engine.py            # Genetic Algorithm core logic
├── ga_chromosome.py        # Chromosome representation (image subset + flags)
├── ga_constants.py         # GA hyperparameters
├── camera_calibrator.py    # CameraCalibrator: wraps cv2.calibrateCamera
├── pattern_manager.py      # PatternManager: wraps cv2.ccalib.CustomPattern
├── utils.py                # ChessboardValidator and utilities
└── res/
    └── camera_calib_tool_icon.png
```

---

## 🔧 Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` + `opencv-contrib-python` | Core computer vision, `ccalib` module |
| `PyQt6` | GUI framework |
| `numpy` | Numerical operations |

Install standard dependencies:
```bash
pip install PyQt6 numpy
```

### ⚠️ OpenCV + ccalib on Windows 11 (Python 3.14)

The `cv2.ccalib` module is **not included** in the standard `opencv-python` pip
package. To use it, OpenCV must be compiled from source with the
`opencv_contrib` modules enabled.

**Steps used to integrate ccalib on Windows 11 + Python 3.14.3:**

1. Download [OpenCV](https://github.com/opencv/opencv) and
   [opencv_contrib](https://github.com/opencv/opencv_contrib) sources
2. Use **CMake** to configure the build, enabling only the required modules:
   - `core`, `calib3d`, `ccalib`, `features2d`, `imgproc`, `highgui`
3. Build with Visual Studio (or MSVC)
4. Place the compiled `.pyd` file — rename it to `cv2.pyd` — and all
   required `.dll` files into the Python `site-packages` directory:
```text
C:\Users\<user>\AppData\Local\Programs\Python\Python314\Lib\site-packages\cv2\
```
5. Add or update `__init__.py` in that folder to expose the `cv2` module
   correctly

After this, `import cv2` and `from cv2 import ccalib` work normally in any
Python script or virtual environment pointing to that interpreter.

---

## 🚀 Usage

```bash
python app.py
```

### Real-Time mode
1. Click **▶ Start Camera**
2. Draw the ROI rectangle around the calibration pattern in the video feed
3. Click **✅ Confirm ROI** — the pattern is created
4. Click **📸 Capture Frame** multiple times (≥ 10 recommended) from different angles
5. Click **⚙️ Perform Calibration + GA** — standard calibration runs first,
   then the GA optimizer starts automatically in background

### Offline mode
1. Click **📂 Load Pattern** — select your pattern image
2. Click **🗂️ Load Calibration Images** — select a batch of photos
3. Click **⚙️ Run Calibration + GA** — the app processes all images,
   runs `calibrateCamera`, then launches the GA automatically

Results are saved to the `out/` folder.

---

## ⚙️ GA Parameters

Tunable in `ga_constants.py`:

| Parameter | Default | Description |
|---|---|---|
| `GA_EPOCHS` | 25 | Number of evolution generations |
| `GA_POOL_SIZE` | 50 | Population size |
| `GA_P_MUTATION` | 0.25 | Mutation probability |
| `GA_P_CROSSOVER` | 0.75 | Crossover probability |
| `GA_T_CATACLYSM` | 0.01 | Diversity threshold triggering cataclysm |
| `GA_P_CATACLYSM` | 0.50 | Fraction of population replaced on cataclysm |
| `MIN_GENES` | 5 | Minimum active images per chromosome |
| `OPTIMIZE_FLAGS` | True | Whether to also optimize calibration flags |
| `PARALLEL` | True | Run fitness evaluations in parallel threads |

---

## 💡 Inspiration & References

- **GA calibration image selection**:
  [ga_stereocalib](https://github.com/asergiu/ga_stereocalib) by Alexandru-Ion Marinescu and A. Sergiu
  [IJCAI 2021 AI4AD Workshop](https://www.youtube.com/watch?v=Jyu6Z_Fc5pE) by Alexandru-Ion Marinescu, Adrian-Sergiu Darabant and Tudor-Alexandru Ileni
  genetic algorithm for stereo camera calibration image subset selection.
  This project adapts and extends that approach to monocular calibration with
  custom patterns.

- **Custom Pattern Calibration**:
  [OpenCV ccalib tutorial (YouTube)](https://www.youtube.com/watch?v=Cm4YbwL43Ww)
  and the official
  [OpenCV ccalib module](https://docs.opencv.org/4.x/d3/d44/group__ccalib.html)
  (`cv2.ccalib.CustomPattern`), part of `opencv_contrib`.

---

## 📄 License

This project is intended for research and educational use.