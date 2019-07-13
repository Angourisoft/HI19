from hypo2.base.basef import BaseHIObj
import hypo2.model
from hypo2.model import HIModel
from hypo2.preprocessor import Preprocessor
from hypo2.dataset import Dataset
import os
import PIL.ImageOps
import numpy as np
from hypo2.base.cache import Cache
from hypo2.addit.functions import Functional as F

class RunEnv(BaseHIObj):
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self.model = HIModel(cfg)
        self.preprocessor = Preprocessor(cfg)

    def get_center(self, image):
        normed = self.preprocessor.norm(image)
        words = self.preprocessor.segment_words(normed)
        return self.model.get_center(F.words2word_block(words)), len(words)

    def open_image(self, path):
        return self.preprocessor.open(path)

    def differ(self, im1, im2):
        return self.dist(self.get_center(im1)[0], self.get_center(im2)[0])

    def dist(self, c1, c2):
        return F.dist(c1, c2)

    def differ_from_paths(self, path1, path2):
        return self.differ(self.open_image(path1), self.open_image(path2))

class FitEnv(BaseHIObj):
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self.model = HIModel(self.cfg)

    def fit(self, dataset_paths, verbose=True, plot=False):
        ds = Dataset(self.cfg)
        X, y = ds.gen_dataset(dataset_paths, verbose=verbose)
        self.model.fit(X, y, verbose=verbose, plot=plot)
        return self.model

    def save(self):
        self.model.save()

    def gen_paths(self, path):
        return F.gen_paths(path)


def clear_cache(cfg):
    chc = Cache(cfg)
    return chc.clear()
