import time
import ipdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt


from .context.tracklets.tps import TPS, makeL, liftPts

# source control points
x, y = np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)
x, y = np.meshgrid(x, y)
xs = x.flatten()
ys = y.flatten()
cps = np.vstack([xs, ys]).T

# target control points
xt = xs + np.random.uniform(-0.3, 0.3, size=xs.size)
yt = ys + np.random.uniform(-0.3, 0.3, size=ys.size)

# dense grid
N = 30
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
x, y = np.meshgrid(x, y)
xgs, ygs = x.flatten(), y.flatten()
gps = np.vstack([xgs, ygs]).T


#Control
# construct L
L = makeL(cps)

# solve cx, cy (coefficients for x and y)
xtAug = np.concatenate([xt, np.zeros(3)])
ytAug = np.concatenate([yt, np.zeros(3)])
cx = nl.solve(L, xtAug) # [K+3]
cy = nl.solve(L, ytAug)
# transform
pgLift = liftPts(gps, cps) # [N x (K+3)]
xgt = np.dot(pgLift, cx.T)
ygt = np.dot(pgLift, cy.T)

# TPS
pt = TPS(cps, np.vstack([xt, yt]).T, gps)

# display
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.subplot(1, 3, 1)
plt.title('Source')
plt.grid()
plt.scatter(xs, ys, marker='+', c='r', s=40)
plt.scatter(xgs, ygs, marker='.', c='r', s=5)
plt.subplot(1, 3, 2)
plt.title('Target')
plt.grid()
plt.scatter(xt, yt, marker='+', c='b', s=40)
plt.scatter(pt[:, 0], pt[:, 1], marker='.', c='b', s=5)
plt.subplot(1, 3, 3)
plt.title('Control')
plt.grid()
plt.scatter(xt, yt, marker='+', c='b', s=40)
plt.scatter(xgt, ygt, marker='.', c='b', s=5)
plt.show()