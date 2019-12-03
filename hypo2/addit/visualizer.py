import random
import torch
from hypo2.base.basef import BaseHIObj
from hypo2.model import HIModel
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from hypo2.addit.functions import Functional as F

class Visualizer(BaseHIObj):
    def __init__(self, config):
        self.config = config.copy()

    def get_centers_from_xy(self, X, y, classes=None, model=None):
        if model is None:
            model = HIModel(self.config)
        model.train(False)
        cfg = self.config.copy()
        vX_tr, vy_tr = F.prepare_ds(X, y, cfg, False)

        if classes is None:
            vX_t, vy_t = vX_tr, vy_tr
        else:
            vX_t, vy_t = [], []
            for i in range(len(vX_tr)):
                if vy_tr[i] in classes:
                    vX_t.append(vX_tr[i])
                    vy_t.append(vy_tr[i])
            vX_t = torch.stack(vX_t)
            vy_t = torch.stack(vy_t)

        fXl = []
        fyl = []
        for i in tqdm(range(len(vy_t))):
            rad = random.randint(0, len(vX_t) - self.config.BATCH_SIZE)
            if True:
                x = vX_t[rad : rad + self.config.BATCH_SIZE]
                p3 = F.runtime_preprocess(self.config, x)
                fXl.extend(model.extract(p3).detach())
                fyl.extend(vy_t[rad : rad + self.config.BATCH_SIZE])
        return fXl, fyl

    def build_comp(self, fXl, fyl, comp):
        cm = comp(n_components=2)
        vecs = cm.fit_transform([l.tolist() for l in fXl])
        onearr = []
        for i in range(len(vecs)):
            onearr.append((vecs[i], fyl[i]))
        colors = sorted(onearr, key=lambda x: x[1])
        diffars = [list() for i in range(colors[-1][1] + 1)]
        for col in colors:
            diffars[col[1]].append(col[0])
        for diffarr in diffars:
            plt.scatter([v[0] for v in diffarr], [v[1] for v in diffarr])
        plt.show()

    def get_distance_distribution(self, fXl, fyl):
        the_same_person = []
        different_people = []
        for _ in range(1000):
            i1 = random.randint(0, len(fXl) - 1)
            i2 = random.randint(0, len(fXl) - 1)
            if i1 == i2:
                continue
            d = ((fXl[i1] - fXl[i2]) ** 2).sum().item()
            if fyl[i1] == fyl[i2]:
                the_same_person.append(d)
            else:
                different_people.append(d)
        return the_same_person, different_people

    def build_dist_distr(self, the_same_person, different_people):
        sns.distplot(the_same_person, label="one")
        sns.distplot(different_people, label="two")
        plt.legend()
