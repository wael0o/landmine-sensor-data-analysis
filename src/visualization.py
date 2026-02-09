import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_clusters(X, labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for lab in np.unique(labels):
        mask = labels == lab
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            X[mask, 2],
            label=str(lab),
            s=40,
            alpha=0.8
        )

    ax.set_xlabel("V")
    ax.set_ylabel("H")
    ax.set_zlabel("S")
    ax.set_title(title)
    ax.legend()
    plt.show()
