from tqdm import tqdm_notebook as tqdm
from hypo2.addit.functions import Functional as F
from hypo2.preprocessor import Preprocessor
import os
import random
import matplotlib.pyplot as plt


def countacc(X, y, thr):
    s = 0
    for i in enumerate(X):
        if (X[i] < thr and y[i] == 0) or (X[i] > thr and y[i] == 1):
            s += 1
    return s / len(X)

def validate(config, model, paths, PAX, verbose=True):
    valpr = Preprocessor(config)
    pointss = []
    for cpaths in tqdm(paths):
        w = []
        for p in cpaths:
            try:
                w.extend(valpr.open_norm_segm(p))
            except:
                pass
        if len(w) < 2 * PAX:
            if verbose:
                print(os.path.dirname(cpaths[0]), "missed, not enough words:", len(w))
            continue
        pointss.append([])
        for i in range(len(w) // PAX):
            pointss[-1].append(model.get_center(F.words2word_block(w[i * PAX: (i + 1) * PAX])))



    X_dists = []
    y_dists = []
    for pss in tqdm(pointss):
        for i in range(len(pss)):
            for j in range(i + 1, len(pss)):
                X_dists.append(F.dist(pss[i], pss[j]).item())
                y_dists.append(0)



    zero_count = len(y_dists)
    for i in range(zero_count):
        random.shuffle(pointss)
        s1 = pointss[0]
        s2 = pointss[1]
        ps1 = random.choice(s1)
        ps2 = random.choice(s2)
        X_dists.append(F.dist(ps1, ps2).item())
        y_dists.append(1)



    allnums = sorted(X_dists)
    ns = []
    bacc, bh = 0, 0
    rctr = range(0, len(allnums) - 1, max(300 // (PAX ** 2), 1))
    if verbose:
        rctr = tqdm(rctr)
    XXX = []
    for i in rctr:
        thr = (allnums[i] + allnums[i + 1]) / 2
        XXX.append(thr)
        acc = countacc(X_dists, y_dists, thr)
        ns.append(acc)
        if acc > bacc:
            bacc = acc
            bh = thr
    if verbose:
        plt.plot(XXX, ns)


    if verbose:
        print("Best accuracy:", bacc)
        print("Best threshold:", bh)

    return bh

def test_project(config, model, paths):
    thrs = []

    for pax in tqdm([1, 2, 10, 25, 70, 200]):
        thrs.append(validate(config, model, paths, pax, False))

    print("Thresholds:", thrs)
