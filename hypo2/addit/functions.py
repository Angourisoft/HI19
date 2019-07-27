import os
import random
import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import math

REVE = 1 - 1 / math.e

class Functional:
    @staticmethod
    def valid_path(path):
        if type(path) != str:
            return False
        return os.path.exists(path)

    @staticmethod
    def add_padding_4(res, pattern):
        before_0 = -(res.shape[3] - pattern[0]) // 2
        after_0 = -(res.shape[3] - pattern[0]) // 2 + (res.shape[3] - pattern[0]) % 2
        before_1 = -(res.shape[2] - pattern[1]) // 2
        after_1 = -(res.shape[2] - pattern[1]) // 2 + (res.shape[2] - pattern[1]) % 2
        return np.pad(res, [(0, 0), (0, 0), (before_1, after_1), (before_0, after_0)], mode="constant", constant_values=1)

    @staticmethod
    def runtime_preprocess(config, bX):
        dev = torch.device(config.DEVICE)
        bX = torch.tensor(Functional.add_padding_4(1 - (bX + 1/2), (config.NN_INPUT_SIZE[0], config.NN_INPUT_SIZE[1])))
        tnoise = torch.from_numpy(np.stack([np.stack([np.random.randn(config.NN_INPUT_SIZE[0], config.NN_INPUT_SIZE[1])] * 3, axis=2) / 30 for j in range(bX.shape[0])])).transpose(1, 3)
        return (1 - (bX.type(torch.float) + tnoise.type(torch.float)).to(dev))

    @staticmethod
    def count_distr(fy, cls):
        PS = [0 for i in range(cls)]
        for y in fy:
            PS[y.item()] += 1
        return PS

    @staticmethod
    def safe_path(path):
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    @staticmethod
    def get_x_classes(X, y, cls):
        X_classes = [[] for i in range(cls)]
        for f in range(len(y)):
            X_classes[y[f]].append(X[f])
        return X_classes

    @staticmethod
    def get_ds(X_classes, length, config, todel=False):
        fX = []
        fy = []
        for i in range(length):
            class_id = random.randint(0, config.CLASS_COUNT - 1)
            if len(X_classes[class_id]) > 0:
                dd = random.randint(0, len(X_classes[class_id]) - 1)
                fX.append(X_classes[class_id][dd])
                fy.append(class_id)
                if todel:
                    del X_classes[class_id][dd]
        fX = torch.from_numpy(np.stack(fX).transpose((0, 3, 1, 2)).astype(np.float))
        fy = torch.from_numpy(np.stack(fy))
        return fX, fy

    @staticmethod
    def prepare_ds(X, y, cfg, test=True):
        len1 = round(len(y) * (1 - cfg.VAL_SHARE))
        len2 = round(len(y) * cfg.VAL_SHARE / REVE)
        X_classes = Functional.get_x_classes(X, y, cfg.CLASS_COUNT)
        assert len(X_classes) == cfg.CLASS_COUNT, "An error occurred while get X_classes"
        if test:
            fX_train, fy_train = Functional.get_ds(X_classes, len1, cfg, True)
            fX_test, fy_test = Functional.get_ds(X_classes, len2, cfg, False)
            return fX_train, fy_train, fX_test, fy_test
        else:
            return Functional.get_ds(X_classes, len1 + len2, cfg, False)

    @staticmethod
    def gen_paths(path):
        dirs = os.listdir(path)
        res = []
        for dir in dirs:
            pp = path + "/" + dir
            files = os.listdir(pp)
            res.append([pp + "/" + i for i in files])
        return res

    @staticmethod
    def validate_model(config, model, fX_test, fy_test):
        all = 0
        s = 0
        for i in range(config.VAL_EPOCHS):
            batch_id = random.randint(0, len(fX_test) - config.BATCH_SIZE)
            X_b = Functional.runtime_preprocess(config, fX_test[batch_id: batch_id + config.BATCH_SIZE])
            ytrue = fy_test[batch_id : batch_id + config.BATCH_SIZE]
            ypred = model(X_b)
            all += config.BATCH_SIZE
            s += (torch.argmax(ypred.cpu(), dim=1) == ytrue.type(torch.long)).sum().item()
        acc = s / all
        return acc

    @staticmethod
    def words2word_block(words):
        if type(words) == list:
            words = np.stack(words)
        x = words.transpose((0, 3, 2, 1))
        return x

    @staticmethod
    def dist(c1, c2):
        return ((c1 - c2) ** 2).sum() ** 0.5
