import numpy as np
import matplotlib.pyplot as plt


def visualize_network(psm=None, nodes=(), xfmrs=()):
    if psm is None:
        psm = read_psm()
    node_xy = list(zip(*[(n.X_coord, n.Y_coord) for n in psm.Loads]))
    fig = plt.figure(figsize=(6, 6))
    h = plt.scatter(*node_xy, s=5) # plot all loads for context
    for _node in nodes:
        n = psm.Node_Dict[_node]
        plt.scatter(n.X_coord, n.Y_coord, c="g", s=5)

    #for tfname in ['E72805103080326', 'E72805103097973', 'E72805203078706', 'E72805203078859', 'E72805203079531', ]:
    tfs = [psm.Branch_Dict[tfname] for tfname in xfmrs]
    X, Y = list(zip(*[(tf.X_coord, tf.Y_coord) for tf in tfs]))
    color = np.linspace(0, 1, len(tfs))
    plt.scatter(X, Y, c=color, cmap="Wistia", edgecolor="k", s=30, lw=0.5)
    #for tfname in xfmrs:
    #    tf = psm.Branch_Dict[tfname]
    #    plt.scatter(tf.X_coord, tf.Y_coord, c="magenta", edgecolor="k", s=12)
    

def visualize_pf_result(psm, node_df, branch_df, fileprefix=""):
    nodenames = [n.name for n in psm.Nodes]
    branchnames = [b.name for b in psm.Branches]
    branch_dict = {b.name: b for b in psm.Branches}

    ndf = node_df[[n in nodenames for n in node_df["name"]]]
    bdf = branch_df[[n in branchnames for n in branch_df["name"]]]

    fig = plt.figure(figsize=(6, 6))
    V = list(map(np.amax, ndf.V.values))
    res = 0.5
    h = plt.scatter(ndf.X, ndf.Y, s=0.4, c=V, cmap="coolwarm", vmin=1-res, vmax=1+res)
    cbar = plt.colorbar(h)
    
    branch_ends = [((b.X_coord, b.X2_coord), (b.Y_coord, b.Y2_coord)) for b in psm.Branches]
    for x, y in branch_ends:
        plt.plot(x, y, marker=None, c="y", lw=0.1, ls=":")

    tfs = bdf[bdf["type"] == "transformer"]
    S = tfs["S"].values
    for i, tfname in enumerate(tfs["name"].values):
        S[i] *= psm.Sbase_1ph * 1e-3 / branch_dict[tfname].ratedKVA
    #S = np.sum(tfs["I"].values*tfs["V"].values, axis=1) # these are magnitudes, no angles
    #S = np.array([np.sum(i*v) for i, v in zip(tfs["I"].values, tfs["V"].values)])
    inds = S > 0.5
    plt.scatter(tfs.X[inds], tfs.Y[inds], marker="s", c=S[inds], cmap="coolwarm", vmin=1-res, vmax=1+res)

    plt.xticks([])
    plt.yticks([])
    plt.savefig(fileprefix+"result_map.pdf", bbox_inches="tight")
    plt.show()
            

def read_psm():
    #import sys
    #sys.path.append("../MAPLE_BST_Demo/")
    import pickle
    with open("data/Alburgh/Alburgh_Model.pkl", "rb") as f:
        psm = pickle.load(f)
    return psm

if __name__ == "__main__":
    visualize_network()
    plt.show()
