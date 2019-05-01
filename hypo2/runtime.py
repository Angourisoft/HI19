from hypo2.basef import BaseHIObj
import hypo2.model
from hypo2.model import HIModel
from hypo2.normalization import Normalizator
from hypo2.preprocessing import WordSegmentator
from hypo2.dataset import Dataset

class RunTime(BaseHIObj):
    def __init__(self, cfg, model, word_segmentator, normalizer):
        self.cfg = cfg
        self.model = model
        self.word_segmentator = word_segmentator
        self.normalizer = normalizer

    def classify_proba(self, image):
        normed = self.normalizer.norm(image)
        words = self.word_segmentator.segment_words(normed)
        return self.model.predict_proba(words)

    def classify(self, image):
        return self.classify_proba(image).argmax()

    def differ(self, im1, im2):
        return ((self.classify_proba(im1) - self.classify_proba(im2)) ** 2).sum()

class FitTime(BaseHIObj):
    def __init__(self, cfg, dataset_paths):
        self.cfg = cfg
        self.paths = dataset_paths

    def fit(self, verbose=True, plot=False):
        config = self.cfg
        normalizator = Normalizator(config)
        word_segmentator = WordSegmentator(config)
        ds = Dataset(config, normalizator, word_segmentator)
        X, y = ds.gen_dataset(self.paths)
        model = HIModel(self.cfg)
        model.fit(X, y, verbose=verbose, plot=plot)
        return model, normalizator, word_segmentator
