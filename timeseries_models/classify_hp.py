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
    if os.path.exists("M.npy"):
        S = np.load("M.npy")
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
        #S = marginals(ami)
        #np.save("M.npy", S)
        
    print(y.dtype, y.shape, y.sum())
    print(S.dtype, S.shape)


def train_model(S, y):
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

    tree = clf.tree_
    print("tree")
    print("depth", tree.max_depth)
    print("nodes", tree.node_count)


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
    def __init__(self):
        self.t0 = time.perf_counter()

    def print(self, s):
        t = time.perf_counter() - self.t0
        print(s, round(t, 2), flush=True)

    def reset(self):
        self.t0 = time.perf_counter()

if __name__ == "__main__":
    run()
