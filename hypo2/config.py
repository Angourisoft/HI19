from hypo2.basef import BaseConfig


class RunConfig(BaseConfig):
    VERT_RAY_THRESHOLD = 35
    VERT_RAY_CHUNKMINSIZE = 10
    VERT_RAY_CHUNKMAXSIZE = 45
    HORIZ_RAY_THRESHOLD = 0
    HORIZ_RAY_CHUNKMINSIZE = 65
    HORIZ_RAY_CHUNKMAXSIZE = 180
    FINAL_SIZE = (HORIZ_RAY_CHUNKMAXSIZE + 20, VERT_RAY_CHUNKMAXSIZE + 10)

    CLASS_COUNT = 2
    BATCH_SIZE = 10
    LAYER_COUNT = 5
    N_EPOCHS = 100
    VAL_SHARE = 0.25
    LEARNING_RATE = 5.0e-6
    DEVICE = "cuda"

    ESTIMATORS_COUNT = 100
    MAX_DEPTH = 6

    FEATURES_COUNT = 300
