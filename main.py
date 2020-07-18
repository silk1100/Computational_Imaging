import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from scipy.stats import multivariate_normal


def myMultiVariateGauss(means=[0,0], stds=[0,0], steps=[0.01, 0.01], corr=False, size=(0,0)):
    xmin = -size[0]//2+1
    xmax = size[0]//2+1
    ymin = -size[1]//2+1
    ymax = size[1]//2+1

    meux = means[0];
    meuy = means[1];
    sx = stds[0];
    sy = stds[1];
    if corr:
        c = 1
    else:
        c = 0
    cov = np.array([
        [sx ** 2, sx*sy*c],
        [sx*sy*c, sy ** 2]
    ])
    xs = np.arange(xmin, xmax + 1, steps[0])
    ys = np.arange(ymin, ymax + 1, steps[1])

    X, Y = np.meshgrid(xs, ys)
    Z = np.empty(X.shape + (2,))
    Z[:, :, 0] = X
    Z[:, :, 1] = Y
    print(X)
    print(Y)
    const = (1.0) / np.sqrt(2.0 * np.pi * np.linalg.det(cov))
    e = np.empty_like(X)
    for r in range(X.shape[0]):
        for c in range(Y.shape[1]):
            e[r, c] = const * np.exp(-0.5 * (
                np.matmul(np.matmul(Z[r, c, :].T, np.linalg.inv(cov)),
                          Z[r, c, :]))
                                     )
    return X, Y, e


def gauss2Dfrom1D(m, s, lower_upper_limit=(0, 0), npoints=100):
    xs = np.linspace(lower_upper_limit[0],lower_upper_limit[1]+1,npoints)
    ys = np.linspace(lower_upper_limit[0],lower_upper_limit[1]+1,npoints)

    X, Y = np.meshgrid(xs, ys)

    g1 = np.random.normal(m,s,npoints)
    gg = g1.reshape(npoints,1)
    _2Dg = np.matmul(gg, gg.T)
    return X, Y, _2Dg


def main():
    xmin = -5
    xmax = 5
    ymin = -5
    ymax = 5

    meux=1;meuy=2;sx=2;sy=5;

    X, Y, g2d = gauss2Dfrom1D(meux, sx, (-5,5), 100)
    X, Y, e = myMultiVariateGauss([meux, meuy], [sx, sy], [0.01, 0.01], True, (10,15))
    print(g2d.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, e, cmap='jet')
    ax.set_xlabel('X-axis')
    ax.set_xlim([-10, 10])
    ax.set_ylabel('Y-axis')
    ax.set_ylim([-10, 10])
    ax.set_zlabel('Z-axis')
    ax.set_zlim([0,1.1*np.max(e)])
    plt.show()


if __name__ == '__main__':
    main()