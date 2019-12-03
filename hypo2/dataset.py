from hypo2.base.basef import BaseHIObj
import numpy as np
from hypo2.base.cache import Cache
from IPython.display import clear_output
from hypo2.preprocessor import Preprocessor
from hypo2.addit.functions import Functional as F

class Dataset(BaseHIObj):
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self.preprocessor = Preprocessor(cfg)

    def gen_dataset(self, paths, verbose=False):     # [ ["a", "b", "c"], ["d", "e"] ]
        assert len(paths) == self.cfg.CLASS_COUNT, "Paths count must be equal to CLASS_COUNT (check cfg param)"
        words = [[] for i in range(len(paths))]
        cache = Cache(self.cfg)
        try:
            for class_id in range(self.cfg.CLASS_COUNT):
                fff = 0
                for path in paths[class_id]:
                    fff += 1
                    if verbose:
                        clear_output(True)
                        print(round(class_id / self.cfg.CLASS_COUNT * 100, 2), "%", round(fff / len(paths[class_id]) * 100, 2), "%")
                    if path in cache["dataset"]:
                        wordsall = cache["dataset"][path]
                    else:
                        wordsall = self.preprocessor.open_norm_segm(path)
                    if self.cfg.DS_MIN_WORDS_PER_PAGE <= len(wordsall):
                        words[class_id].extend(wordsall)
                    cache["dataset"][path] = wordsall
        except:
            print("Memory error!")
        if verbose:
            clear_output(True)
            print("100 %")
        cache.reset()
        y = []
        for w in range(len(words)):
            y.extend([w for i in range(len(words[w]))])
        X = []
        for w in words:
            X.extend(w)
        return np.stack(X), np.array(y)

    def gen_paths(self, path):
        return F.gen_paths(path)
