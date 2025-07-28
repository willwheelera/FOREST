import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import os
import sys
sys.path.append("../")
from device_data import get_meter_data
import clean_data
import time
import tensorly.decomposition as td

def run():
    timer = Timer()
    y = np.load("y.npy")
    datafile = "M13.npy"
    if os.path.exists(datafile):
        S = np.load(datafile)
    else:
        fname_ami = "../data/Alburgh/2024-01-01_2024-12-31_South_Alburgh_Load_corrected.parquet"
        ami = pd.read_parquet(fname_ami)
        #ami = pd.read_parquet("test_sample100.parquet")
        ami = clean_data.clean_all(ami)
        timer.print("ami loaded")

        S = ami.values.T # full data
        
        #meterdf, keys = get_meter_data()
        #cchp_meters = meterdf["Meter Number"][meterdf["cchp"]]
        #y = ami.columns.astype(str).isin(cchp_meters)
        #timer.print("meters loaded")
        #np.save("y.npy", y)

        #S = parafac(ami)
        #s_y = np.concatenate((S, y[:, np.newaxis]), axis=1)
        #np.save("S_y.npy", s_y)
        
        data = ami.values[:8736].reshape(13, 28, 24, -1)
        S = data.mean(axis=1).reshape(13*24, -1).T
        #S = max_marginals(ami)
        
        np.save(datafile, S)
        
    print(y.dtype, y.shape, y.sum())
    print(S.dtype, S.shape)

    plot_by_class(S, y)
    plot_loads(S, y)

def plot_by_class(S, y):
    keep = S.max(axis=1) > 0.1
    print("discarding", (~keep).sum())
    S = S[keep]
    y = y[keep]

    has_pv = S.min(axis=1) < 0
    has_hp = y > 0
    gens = np.where(has_pv)
    S_ = S.reshape(-1, 13, 24)
    S = S_.mean(axis=1)
    toosmall = np.where(S[:, 11] < 1e-6)
    print("gens", len(gens[0]))
    print("hp", has_hp.sum())
    print("gen and hp?", has_hp[gens].sum())
    print("where too small", toosmall[0])
    print(S[toosmall, 11])
    data = {}
    data["hp"] = S[has_hp & ~has_pv].mean(axis=0)
    data["pv"] = S[~has_hp & has_pv].mean(axis=0)
    data["no"] = S[~(has_hp | has_pv)].mean(axis=0)
    data["both"] = S[(has_hp & has_pv)].mean(axis=0)
    for k, v in data.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.show()
    

def plot_loads(S, y):
    s_ = S.reshape(-1, 13, 24)
    s_  = s_ / (np.abs(s_[:, [5]]) + 1)
    #S = s_.reshape(-1, 13*24)

    inds = np.lexsort((S.max(axis=1), y))

    #train_model(S, y)
    plt.plot(np.amax(S[inds], axis=1))
    plt.plot(y[inds])
    plt.xlabel("meters")
    plt.ylabel("max hourly load")# div by May")
    plt.show()

def train_model(S, y):
    timer = Timer()
    w = len(y) / (2*y.sum())
    w_ = len(y) / (2*(~y).sum())
    print(w, w_)
    cw = {True: w, False: w_}

    train_inds = np.random.random(len(y)) < 0.9
    S_t, y_t = S[train_inds], y[train_inds]
    clf = sk.linear_model.LogisticRegression(solver="liblinear", max_iter=800, class_weight=cw).fit(S_t, y_t)
    #clf = sk.tree.DecisionTreeClassifier(class_weight=cw).fit(S[train_inds], y[train_inds])
    #clf = sk.linear_model.Perceptron(max_iter=1000, class_weight=cw).fit(S, y)
    #clf = sk.linear_model.LinearRegression().fit(S, y)
    timer.print("fit model")

    test_inds = ~train_inds
    print("positive test samples?", y[test_inds].sum())

    inds = y>0
    pred = clf.predict(S)
    print("true positive?", np.sum(y[inds] == pred[inds]), "/", inds.sum())
    print("true negative?", np.sum(y[~inds] == pred[~inds]), "/", (~inds).sum())
    pass_test = y[test_inds] == pred[test_inds]
    print("true test?", np.sum(pass_test), "/", (test_inds).sum())
    print("true pos test?", np.sum(pass_test & y[test_inds]), "/", (y[test_inds]).sum())
    print("true neg test?", np.sum(pass_test & (~y[test_inds])), "/", (~y[test_inds]).sum())
    #proba = clf.predict_proba(S[inds])

    print(clf.score(S, y))

    #tree = clf.tree_
    #print("tree")
    #print("depth", tree.max_depth)
    #print("nodes", tree.node_count)


def parafac(df):
    timer = Timer()
    data = df.values[:8736].reshape(52, 7, 24, -1)
    weights, factors = td.parafac(data, rank=6, n_iter_max=10, init='svd', normalize_factors=False)
    timer.print("parafac done")
    W, D, H, S = factors
    #vals = np.einsum("i,wi,di,hi,ni->wdhn", weights, W, D, H, S)
    return S

def marginals(df):
    data = df.values[:8736].reshape(52, 7, 24, -1)
    W = data.sum(axis=(1, 2))
    D = data.sum(axis=(0, 2))
    H = data.sum(axis=(0, 1))
    v = np.concatenate((W, D, H), axis=0)
    return v.T
    
def max_marginals(df):
    data = df.values[:8736].reshape(52, 7, 24, -1)
    W = data.max(axis=(1, 2))
    D = data.max(axis=(0, 2))
    H = data.max(axis=(0, 1))
    v = np.concatenate((W, D, H), axis=0)
    return v.T
    

def svd(ami):
    timer = Timer()
    U, S, Vh = np.linalg.svd(ami.values.astype(float))
    timer.print("svd time")
    

def logit_regression(ami):
    timer = Timer()
    meterdf, keys = get_meter_data()
    timer.print("meters loaded")
    cchp_meters = meterdf["Meter Number"][meterdf["cchp"]]
    y = ami.index.astype(str).isin(cchp_meters)
    print(y, y.sum())

    clf = sk.linear_model.LogisticRegression().fit(ami.values, y)
    timer.print("fit model")

    print(y[:10].astype(int))
    print(clf.predict_proba(ami.values[:10]))
    print(clf.predict(ami.values[:10]))
    print(clf.score(ami.values, y))

    timer.print("computed results")

def remove_zeros(df, threshold=1):
    select = np.abs(df.values).sum(axis=1) > threshold
    return df.loc[select]

class Timer:
    def __init__(self, fill=0):
        self.t0 = time.perf_counter()
        self.fill = fill

    def print(self, s):
        t = time.perf_counter() - self.t0
        print(s.ljust(self.fill), round(t, 2), flush=True)

    def reset(self):
        self.t0 = time.perf_counter()

if __name__ == "__main__":
    run()
