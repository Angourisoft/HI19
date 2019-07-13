from hypo2.base.basef import BaseConfig


class RunConfig(BaseConfig):

    # PREPROCESSOR
    DS_MIN_WORDS_PER_PAGE = 30                                              # number
    VERT_RAY_THRESHOLD = 19                                                 # number, "adaptive"
    VERT_RAY_CHUNKMINSIZE = 10                                              # number
    VERT_RAY_CHUNKMAXSIZE = 45                                              # number
    HORIZ_RAY_THRESHOLD = 0                                                 # number, "adaptive"
    HORIZ_RAY_CHUNKMINSIZE = 65                                             # number
    HORIZ_RAY_CHUNKMAXSIZE = 180                                            # number
    SHEET_ANGLE = 0                                                         # "adaptive", "0"
    FINAL_SIZE = (HORIZ_RAY_CHUNKMAXSIZE + 20, VERT_RAY_CHUNKMAXSIZE + 10)  # tuple of numbers
    NN_INPUT_SIZE = (224, 224)                                              # tuple of numbers

    # FITTING
    CLASS_COUNT = 33                                                        # number
    BATCH_SIZE = 10                                                         # number
    LAYER_COUNT = 5                                                         # number
    N_EPOCHS = 100000                                                       # number
    VAL_PERIOD = 4                                                          # number
    VAL_SHARE = 0.25                                                        # floating number
    VAL_EPOCHS = 4                                                          # number
    LEARNING_RATE = 5.0e-6                                                  # floating number
    DEVICE = "cuda"                                                         # "cuda", "cpu"
    DS_SHUFFLE_LEN = 3000
    BACKUP_PERIOD = 500                                                    # number

    # VISUALIZATION
    PLOT_REDRAW_PERIOD = 10                                                 # number
    PLOT_REDRAW_DENSE = 1                                                   # number
    SMOOTH_POWER = 30                                                       # number
    DS_VISUALIZE_LEN = 500

    # PATHS
    CACHE_PATH = "./hi19media/cache"  # string, None
    BACKUP_DIRECTORY = "D:/main/ml_prj/SchoolHWA/model_backups/"  # string, "None"
    MODEL_PATH = "D:/main/ml_prj/SchoolHWA/mainm.h5"

    # GENERAL OPTIONS
    FEATURES_COUNT = 300                                                    # number
