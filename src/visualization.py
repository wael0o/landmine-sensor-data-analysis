import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_clusters(X, labels, title):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=labels
    )

    ax.set_xlabel("V")
    ax.set_ylabel("H")
    ax.set_zlabel("S")
    ax.set_title(title)

    plt.show()