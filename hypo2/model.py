from hypo2.base.basef import BaseHIObj
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
from torchvision.models.resnet import ResNet, Bottleneck
import time
from IPython.display import clear_output
from hypo2.addit.functions import Functional as F

def createresnet(**kwargs):
    model = ResNet(Bottleneck, [4, 12, 46, 4], **kwargs)
    return model

class FeatExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.copy()

        self.i = createresnet(num_classes=config.FEATURES_COUNT)
        if F.valid_path(config.MODEL_PATH):
            try:
                self.i.load_state_dict(torch.load(config.MODEL_PATH))
            except:
                print("An error occured while trying to load weights. A new model created.")

        self.t = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.FEATURES_COUNT, config.CLASS_COUNT),
            nn.Softmax()
        )

    def classify_proba(self, x):
        x = self.extract(x)
        x = self.t(x)
        return x

    def extract(self, x):
        return self.i(x)

class HIModel(BaseHIObj):
    def __init__(self, config):
        self.config = config.copy()
        self.device = torch.device(config.DEVICE)
        self.featextractor = FeatExtractor(self.config).to(self.device)

    def __pr(self, *s):
        if self.__v:
            print(*s)

    def fit(self, X, y, verbose=True, plot=False):
        assert len(X) == len(y), "X and y must have the same size"
        self.__v = verbose
        self.__p = plot
        pr = self.__pr
        cfg = self.config
        X_train, y_train, X_test, y_test = F.prepare_ds(X, y, cfg, True)
        pr("First stage: fitting nn")
        REDRAW_SIZE = 20
        lasttime = time.time()
        ydistr = F.count_distr(y_train)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(1 / np.array(ydistr)).to(self.device).type(torch.float))
        opt = torch.optim.Adam(self.featextractor.parameters(), lr=cfg.LEARNING_RATE)
        losses = []
        accs = []
        g_losses = []
        g_accs = []

        valaccs = []
        self.featextractor.train(False)
        valacc = F.validate_model(cfg, self.featextractor.classify_proba, X_test, y_test)
        self.featextractor.train(True)
        valaccs.append(valacc)

        g_valaccs = valaccs[:]

        self.featextractor.train(True)
        uniqpath = "HI" + str(int(time.time()))

        lastsave = ""
        for i in range(cfg.N_EPOCHS):
            batch_id = random.randint(0, len(X_train) - cfg.BATCH_SIZE)
            X_b = F.runtime_preprocess(self.config, X_train[batch_id: batch_id + cfg.BATCH_SIZE])
            ytrue = y_train[batch_id: batch_id + cfg.BATCH_SIZE]
            ypred = self.featextractor.classify_proba(X_b)
            loss = crit(ypred, ytrue.to(self.device).type(torch.long))
            loss.backward()
            opt.step()
            opt.zero_grad()

            all = cfg.BATCH_SIZE
            s = torch.argmax(ypred.cpu(), dim=1) == ytrue.type(torch.long)
            accs.append(s.sum().item() / all)
            losses.append(loss.item())
            g_accs.append(sum(accs[-cfg.SMOOTH_POWER:]) / len(accs[-cfg.SMOOTH_POWER:]))
            g_losses.append(sum(losses[-cfg.SMOOTH_POWER:]) / len(losses[-cfg.SMOOTH_POWER:]))

            if i % REDRAW_SIZE == REDRAW_SIZE - 1:
                clear_output(True)
                plt.figure(figsize=[24, 6.7])
                plt.subplot(1, 2, 1)
                plt.plot(g_losses[::cfg.PLOT_REDRAW_DENSE], label="loss")
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(g_accs[::cfg.PLOT_REDRAW_DENSE], label="train acc")
                plt.plot(g_valaccs[::cfg.PLOT_REDRAW_DENSE], label="val acc")
                plt.legend()
                plt.show()
                pr("acc:", round(sum(g_accs[-1 - REDRAW_SIZE: -1]) / REDRAW_SIZE, 3))
                pr("loss:", round(sum(g_losses[-1 - REDRAW_SIZE: -1]) / REDRAW_SIZE, 3))
                tm = time.time()
                tmgone = tm - lasttime
                lasttime = tm
                pr("last val acc:", round(g_valaccs[-1], 3))
                pr(round(REDRAW_SIZE / tmgone, 2), "epochs per second")
                pr(round(1000 * tmgone / REDRAW_SIZE, 2), "seconds for 1000 epochs")
                if lastsave != "":
                    pr("Last backup is saved to", lastsave)

            if cfg.BACKUP_DIRECTORY is not None and i % cfg.BACKUP_PERIOD == cfg.BACKUP_PERIOD - 1:
                p = cfg.BACKUP_DIRECTORY + uniqpath + "/model_" + str(i)
                self.save(p)
                lastsave = p

            if i % cfg.VAL_PERIOD == cfg.VAL_PERIOD - 1:
                self.featextractor.train(False)
                valacc = F.validate_model(cfg, self.featextractor.classify_proba, X_test, y_test)
                self.featextractor.train(True)
                valaccs.append(valacc)
            else:
                valaccs.append(valaccs[-1])
            g_valaccs.append(sum(valaccs[-cfg.SMOOTH_POWER:]) / len(valaccs[-cfg.SMOOTH_POWER:]))
        return True

    def get_center(self, word_block):
        self.featextractor.train(False)
        r = []
        for w in range(0, len(word_block), self.config.BATCH_SIZE):
            x = F.runtime_preprocess(self.config, word_block[w : w + self.config.BATCH_SIZE])
            r.extend(self.featextractor.extract(x).cpu().detach())
        return sum(r) / len(r)

    def extract(self, x):
        return self.featextractor.extract(x)

    def saveto(self, path=None):
        if path is None:
            path = self.config.MODEL_PATH
        F.safe_path(path)
        torch.save(self.featextractor.i.state_dict(), path)

    def save(self):
        self.saveto()

    def openfrom(self, path=None):
        # This method is deprecated
        if path is None:
            path = self.config.MODEL_PATH
        pathcfg = self.config.copy()
        pathcfg.MODEL_PATH = path
        self.featextractor = FeatExtractor(self.config)

    def open(self):
        self.openfrom()

    def train(self, b):
        self.featextractor.train(b)