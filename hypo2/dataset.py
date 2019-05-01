from hypo2.basef import BaseHIObj
import numpy as np


class Dataset(BaseHIObj):
    def __init__(self, cfg, normalizator, word_segmentator):
        self.cfg = cfg
        self.normalizator = normalizator
        self.word_segmentator = word_segmentator

    def gen_dataset(self, paths):     # [ ["a", "b", "c"], ["d", "e"] ]
        assert len(paths) == self.cfg.CLASS_COUNT, "Paths count must be equal to CLASS_COUNT (check cfg param)"
        words = [[] for i in range(len(paths))]
        for class_id in range(self.cfg.CLASS_COUNT):
            for path in paths[class_id]:
                norm_image = self.normalizator.open_norm(path)
                patterns = self.word_segmentator.segment_words(norm_image)
                words[class_id].extend(patterns)
        y = []
        for w in range(len(words)):
            y.extend([w for i in range(len(words[w]))])
        X = []
        for w in words:
            X.extend([f / 255 for f in w])
        return np.stack(X), np.array(y)
