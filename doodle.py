import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import style


def createGrid(xrange=(0,0), yrange=(0,0), res=(0,0)):
   xmin = xrange[0]
   xmax = xrange[1]
   ymin = yrange[0]
   ymax = yrange[1]
   xres = res[0]
   yres = res[1]
   xs = np.arange(xmin, xmax + xres, xres)
   ys = np.arange(ymin, ymax + yres, yres)

   X, Y = np.meshgrid(xs, ys)

   return X, Y


def stack2Dto3D(X, Y):
   z = np.empty(X.shape+(2,))
   z[:,:,0] = X
   z[:,:,1] = Y

   return z


def multiVariateGauss(meus, stds, dependent=False, stackedPos=np.array([])):
   n_vars = len(meus)
   assert(len(meus)==len(stds))
   assert(len(meus)==stackedPos.shape[2])

   shifted_grid = np.empty_like(stackedPos)
   for ind, m in enumerate(meus):
      shifted_grid[:,:,ind] = stackedPos[:,:,ind]-m

   cov = np.empty((len(meus), len(meus)))
   if dependent:
      k=1
   else:
      k=0
   for r in range(cov.shape[0]):
      cov[r,r] = stds[r]**2

   for r in range(cov.shape[0]):
      for c in range(cov.shape[1]):
            if r!=c:
               cov[r,c] = stds[r]*stds[c]*k

   covinv = np.linalg.inv(cov)
   covdet = np.linalg.det(cov)

   const = (1.0)/(np.sqrt(2*np.pi*covdet))
   e = np.empty_like(stackedPos[:,:,0])
   for x in range(stackedPos.shape[0]):
      for y in range(stackedPos.shape[1]):
         vec = stackedPos[x,y,:]
         e[x, y] = const*np.exp(-0.5*np.matmul(np.matmul(vec.T, covinv), vec))

   return e


def plot3D(X, Y, values):
   fig = plt.figure()
   ax = plt.gca(projection='3d')
   ax.plot_surface(X, Y, values, cmap='jet')
   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.set_zlabel('Z-axis')

   plt.show()


def main():
   print(style.available)
   style.use('Solarize_Light2')
   # return 0
   X, Y = createGrid((-5,5),(-5,5),(0.01, 0.01))
   z = stack2Dto3D(X, Y)

   e = multiVariateGauss([0,0],[1, 3], False, z)

   plot3D(X, Y, e)


if __name__ == '__main__':
    main()