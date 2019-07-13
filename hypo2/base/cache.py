import pickle
import os

class Cache(dict):
    def __init__(self, cfg):
        self.cfg = cfg.copy()
        if (not cfg.CACHE_PATH) or (not os.path.exists(cfg.CACHE_PATH + "/cp")):
            self["dataset"] = {}
            return
        try:
            d = pickle.load(open(cfg.CACHE_PATH + "/cp", "rb"))
            for key in d:
                self[key] = d[key]
        except:
            self["dataset"] = {}
            return

    def __reset(self):
        if self.cfg.CACHE_PATH:
            if not os.path.exists(self.cfg.CACHE_PATH):
                os.makedirs(self.cfg.CACHE_PATH)
            d = {}
            for k in self:
                d[k] = self[k]
            pickle.dump(d, open(self.cfg.CACHE_PATH + "/cp", "wb"))

    def reset(self):
        try:
            self.__reset()
        except:
            self.clear()

    def clear(self):
        if self.cfg.CACHE_PATH and os.path.exists(self.cfg.CACHE_PATH):
            os.remove(self.cfg.CACHE_PATH + "/cp")
            return "Cache cleared"
        else:
            return "Cache not cleared"
