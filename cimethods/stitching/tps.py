import numpy as np
import numpy.linalg as nl
from scipy.spatial.distance import pdist, cdist, squareform

def makeL(cp):
    # cp: [K x 2] control points
    # returns: [(K+3) x (K+3)]
    K = cp.shape[0]
    L = np.zeros((K+3, K+3))
    L[:K, 0] = 1
    L[:K, 1:3] = cp
    L[K, 3:] = 1
    L[K+1:, 3:] = cp.T
    U = squareform(pdist(cp, metric='euclidean'))
    U = U * U
    U[U == 0] = 1 # a trick to make R ln(R) 0
    U = U * np.log(U)
    np.fill_diagonal(U, 0)
    L[:K, 3:] = U
    return L

def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # returns: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, 3+K))
    pLift[:,0] = 1
    pLift[:,1:3] = p
    U = cdist(p, cp, 'euclidean')
    U = U * U
    U[U == 0] = 1
    U = U * np.log(U)
    pLift[:,3:] = U
    return pLift

def TPS(cps, cpt, ps):
    # cps: [K x 2] control points at source
    # cpt: [K x 2] same control points at the target transformation
    # ps: [N x 2] points to approximate at sourse
    # returns: pt: [N x 2] approximated points at target
    L = makeL(cps)

    # find TSP coef
    cxtAug = np.concatenate([cpt[:, 0], np.zeros(3)])
    cytAug = np.concatenate([cpt[:, 1], np.zeros(3)])
    wax = nl.solve(L, cxtAug) # [K+3]  w (K items) and a (3 items) coefficients vector 
    way = nl.solve(L, cytAug)  

    # transform
    pLift = liftPts(ps, cps) # [N x (K+3)]
    xt = np.dot(pLift, wax.T)
    yt = np.dot(pLift, way.T)
    return np.vstack([xt, yt]).T