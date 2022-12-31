import numpy as np
from scipy.spatial.distance import cdist
import time

from tracklets import TPS, Tracklet, to_np

def calc_propagation(np_trls, max_len = np.inf, direction = 'forward'):
  if direction == 'backward':
    map_frame_index = lambda fi: F - 1 - fi
  else:
    map_frame_index = lambda fi: fi
  F, N, _ = np_trls.shape
  prop = np.full((F, N, 2), np.nan)
  pts_to_prop = np.full((N, 2), np.nan)
  pts_to_prop_idx = np.full((N,), False)
  # frame index abstracts frame number for both forward and back- iterations
  for fi in range(1, F): 
    prev_points = np_trls[map_frame_index(fi-1), :, :]
    cur_points = np_trls[map_frame_index(fi), :, :]
    
    prev_idx = ~np.isnan(prev_points[:, 0])
    curr_idx = ~np.isnan(cur_points[:, 0])

    ctrl_idx = np.logical_and(prev_idx, curr_idx)

    new_prop_idx = np.logical_and(prev_idx, ~curr_idx)
    pts_to_prop_idx = np.logical_or(pts_to_prop_idx, new_prop_idx)
    pts_to_prop[new_prop_idx] = prev_points[new_prop_idx]

    max_gap_fi = fi - max_len
    if max_gap_fi > 0:
      prev_mg_idx = ~np.isnan(np_trls[map_frame_index(max_gap_fi-1), :, 0])
      no_curr_mg_idx = np.isnan(np_trls[map_frame_index(max_gap_fi), :, 0])
      mg_stop_idx = np.logical_and(prev_mg_idx, no_curr_mg_idx)
      pts_to_prop_idx[mg_stop_idx] = False

    #print(f, np.count_nonzero(prev_idx), np.count_nonzero(curr_idx), np.count_nonzero(ctrl_idx), np.count_nonzero(new_prop_idx))
    if np.count_nonzero(pts_to_prop_idx) == 0:
      continue

    pts_to_prop[pts_to_prop_idx] = TPS(prev_points[ctrl_idx], cur_points[ctrl_idx], pts_to_prop[pts_to_prop_idx])
    prop[map_frame_index(fi), pts_to_prop_idx] = pts_to_prop[pts_to_prop_idx]
  return prop

def make_phi(tracklets: [Tracklet], gap_max: int, d_max: float): #TODO: fiducials_max: int
  np_trls = to_np(tracklets)
  F, N, _ = np_trls.shape

  # Forward propagation first:
  t0 = time.time() 
  fwd_prop = calc_propagation(np_trls, max_len=gap_max)
  print("Propagated forward", time.time() - t0)

  # Backward propagation then:
  t0 = time.time()
  bwd_prop = calc_propagation(np_trls, max_len=gap_max, direction="backward")
  print("Propagated backward", time.time()-t0)

  phi = np.full((N, N), np.inf)
  def update_phi(src, targ, src_idx, targ_idx):
    d0 = cdist(src[src_idx, :], targ[targ_idx, :])
    d0[d0 > d_max] = np.inf
    idx = np.ix_(src_idx, targ_idx)
    phi[idx] = np.minimum(phi[idx], d0)

  for f in range(0, F):
    start_pts_idx = end_pts_idx = ~np.isnan(np_trls[f, :,0])
    if f < F-1:
      end_pts_idx = np.logical_and(end_pts_idx, np.isnan(np_trls[f+1, :,0]))
    if f > 0:
      start_pts_idx = np.logical_and(start_pts_idx, np.isnan(np_trls[f-1, :,0]))
    fwd_prop_idx = ~np.isnan(fwd_prop[f, :,0])
    bwd_prop_idx = ~np.isnan(bwd_prop[f, :,0])
    update_phi(fwd_prop[f], bwd_prop[f], fwd_prop_idx, bwd_prop_idx)
    update_phi(np_trls[f], bwd_prop[f], end_pts_idx, bwd_prop_idx)
    update_phi(fwd_prop[f], np_trls[f], fwd_prop_idx, start_pts_idx)

    if f %50 == 49: print("Done ", f+1, " frames")

  # TODO: how can we skip counting of those distances?
  inf_idx = np.zeros((N, N), dtype=bool)
  for i, t1 in enumerate(tracklets):
    for j, t2 in enumerate(tracklets):
      if t1.end + gap_max < t2.start:
        inf_idx[i,j] = True
  phi[inf_idx] = np.inf
      
  # print("All done in ", time.time() - t000)
  return phi, fwd_prop, bwd_prop


