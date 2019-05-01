from PIL import Image
import numpy as np
from hypo2.basef import BaseHIObj

class Normalizator(BaseHIObj):
    def __init__(self, cfg):
        self.cfg = cfg

    def open_norm(self, path):
        return self.norm(Image.open(path))

    def norm(self, im):
        return np.array(im).astype(np.float)
