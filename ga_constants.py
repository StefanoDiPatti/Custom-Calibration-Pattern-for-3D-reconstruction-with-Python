import numpy as np

class GAConstants:
    INF = 1E6

    # Image dimensions for calibration (can be changed if needed)
    IMG_WIDTH  = 640
    IMG_HEIGHT = 480

    # GA parameters
    GA_EPOCHS      = 25
    GA_POOL_SIZE   = 50
    GA_P_MUTATION  = 0.25
    GA_P_CROSSOVER = 0.75
    GA_T_CATACLYSM = 0.01
    GA_P_CATACLYSM = 0.50

    # Minimum number of active images for a valid chromosome
    MIN_GENES = 5

    # If True, also optimizes the flags for calibrateCamera
    OPTIMIZE_FLAGS = True

    # If True, runs the fitness workers in parallel (threading)
    PARALLEL = True
