import pytest

import time
import ipdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt


from tracklets.tps import TPS, makeL, liftPts

@pytest.mark.visual
def test_tps():
  # source control points
  x, y = np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)
  x, y = np.meshgrid(x, y)
  xs = x.flatten()
  ys = y.flatten()
  cps = np.vstack([xs, ys]).T

  # target (deformed) control points
  xt = xs + np.random.uniform(-0.4, 0.4, size=xs.size)
  yt = ys + np.random.uniform(-0.4, 0.4, size=ys.size)
  cpt = np.vstack([xt, yt]).T

  # dense grid to test on
  N = 30
  x = np.linspace(-2, 2, N)
  y = np.linspace(-2, 2, N)
  x, y = np.meshgrid(x, y)
  xgs, ygs = x.flatten(), y.flatten()
  ps = np.vstack([xgs, ygs]).T


  # TPS
  pt = TPS(cps, cpt, ps)

  # display
  plt.xlim(-2.5, 2.5)
  plt.ylim(-2.5, 2.5)
  plt.subplot(1, 2, 1)
  plt.title('Source')
  plt.grid()
  plt.scatter(xs, ys, marker='+', c='r', s=40)
  plt.scatter(xgs, ygs, marker='.', c='r', s=5)
  plt.subplot(1, 2, 2)
  plt.title('Target')
  plt.grid()
  plt.scatter(xt, yt, marker='+', c='b', s=40)
  plt.scatter(pt[:, 0], pt[:, 1], marker='.', c='b', s=5)

  plt.show()