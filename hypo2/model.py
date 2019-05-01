from hypo2.basef import BaseHIObj

import torch.nn as nn
import sklearn
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import Sequential
import os
import pickle
import torch
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])


class NNClassifier(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.inp = []
        for i in range(cfg.LAYER_COUNT):
            self.inp.extend(
                [nn.Conv2d(3, 3, kernel_size=3).to(device),
                 nn.BatchNorm2d(3).to(device)])
        self.flatten = Flatten().to(device)
        self.features = nn.Linear(
            (cfg.FINAL_SIZE[0] - 2 * cfg.LAYER_COUNT) * (cfg.FINAL_SIZE[1] - 2 * cfg.LAYER_COUNT) * 3,
            cfg.FEATURES_COUNT).to(device)
        self.final = Sequential(
            nn.Linear(cfg.FEATURES_COUNT, cfg.CLASS_COUNT),
            nn.Softmax()).to(device)

    def forward(self, x):
        for l in self.inp:
            x = l(x)
        x = self.flatten(x)
        feats = self.features(x)
        return feats, self.final(feats)


class HIModel(BaseHIObj):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.l1 = NNClassifier(config, self.device)
        self.l2 = RandomForestClassifier(n_estimators=config.ESTIMATORS_COUNT, max_depth=config.MAX_DEPTH)

    def __pr(self, *s):
        if self.__v:
            print(*s)

    def fit(self, X, y, verbose=True, plot=False):
        assert len(X) == len(y), "X and y must have the same size"
        X = X.transpose((0, 3, 1, 2))
        self.__v = verbose
        self.__p = plot
        pr = self.__pr
        cfg = self.config
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.VAL_SHARE)
        pr("First stage: fitting nn")
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.l1.parameters(), lr=cfg.LEARNING_RATE)
        losses = []
        for i in range(cfg.N_EPOCHS):
            X_ = []
            y_ = []
            for j in range(cfg.BATCH_SIZE):
                id = random.randint(0, len(X_train) - 1)
                X_.append(torch.from_numpy(X_train[id]))
                y_.append(torch.tensor(y_train[id]))
            X_ = torch.stack(X_).type(torch.float).to(self.device)
            true = torch.stack(y_).to(self.device).type(torch.long)
            pred = self.l1(X_)[1]
            loss = crit(pred, true)
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            pr(i + 1, "/", cfg.N_EPOCHS, "loss:", loss.item())
        pr("Secondstage: fitting forest")
        X_train_2 = self.l1(torch.from_numpy(X_train).to(self.device).type(torch.float))[0]
        X_test_2 = self.l1(torch.from_numpy(X_test).to(self.device).type(torch.float))[0]
        X_train_2 = X_train_2.detach().cpu()
        X_test_2 = X_test_2.detach().cpu()
        self.l2.fit(X_train_2, y_train)
        pr("Final accuracy:", (self.l2.predict(X_test_2) == y_test).mean())
        if plot:
            plt.plot(losses)
        return losses, (X_test, y_test)

    def predict_proba(self, words):
        if type(words) not in [np.ndarray, list] or (type(words) == np.ndarray and len(words.shape) < 4):
            words = [words]
        assert len(words) > 0, "Given empty array"
        if words[0].shape[1] == 3:
            X = torch.stack([torch.from_numpy(word) for word in words]).to(self.device)
        else:
            X = torch.stack([torch.from_numpy(word.transpose((2, 0, 1))) for word in words]).to(self.device)
        X = X.type(torch.float)
        if X.max() > 10:
            X /= 255
        feats = self.l1(X)[0].detach().cpu()
        probas = self.l2.predict_proba(feats)
        r = 0
        for proba in probas:
            r += proba
        r /= len(probas)
        return r

    def predict(self, words):
        return self.predict_proba(words).argmax()

    def validate(self, X, y):
        return (self.predict(X) == y).mean()

    def open(self, path):
        if path[-1] != "/":
            path += "/"
        nnpath = path + "l1.h5"
        forpath = path + "l2.pickle"
        self.l1 = torch.load(nnpath)
        self.l2 = pickle.load(open(forpath, "rb"))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if path[-1] != "/":
            path += "/"
        nnpath = path + "l1.h5"
        forpath = path + "l2.pickle"
        torch.save(self.l1, nnpath)
        pickle.dump(self.l2, open(forpath, 'wb'))
        return "OK"
