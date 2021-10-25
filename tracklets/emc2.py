import numpy as np
from operator import attrgetter
import math

from tracklets import TPS

def make_phi(tracklets, gap_max, d_max):
  # tracklets: [N] { 
  #   start: int, # starting frame
  #   points: [K x (x,y)]  
  # }
  N = len(tracklets)
  # Determine frames count:
  F = 0 
  for tr in tracklets:
    F = max(F, tr.start + len(tr.points))
  
  # Forward propagation first:
  tracklets_by_start = sorted(tracklets, key=attrgetter('start'))
  tracklets_by_end = sorted(tracklets, key=attrgetter('end'))

  f_prop_pts = []
  real_idxs_to_f_prop = {}
  f_prop_frames = [np.array([])] # 0-frame propagations init
  S = 0
  E = 0
  control_tracklets = set()
  for f in range(1, F):
    while S < N and tracklets_by_start[ S ].start < f:
      t = tracklets_by_start[ S ]
      control_tracklets.add(t)
      S += 1
    while E < N and tracklets_by_end[ E ].end < f:
      t = tracklets_by_end[ E ]
      control_tracklets.discard(t)
      if t.end == f-1:
        real_idxs_to_f_prop[ t.id ] = len(f_prop_pts)
        f_prop_pts.append(t.points[ len(t.points) - 1 ])
      E += 1
    
    ctrl_p_s = [] # source control points
    ctrl_p_t = [] # target control points
    for t in control_tracklets:
      ctrl_p_s.append(t.point_at(f-1))
      ctrl_p_t.append(t.point_at(f))
    
    if len(f_prop_pts) == 0:
      f_prop_pts_np = np.array([])
    else:
      f_prop_pts_np = TPS(np.array(ctrl_p_s), np.array(ctrl_p_t), np.array(f_prop_pts))
      f_prop_pts = f_prop_pts_np.tolist()
    f_prop_frames.append(f_prop_pts_np)  # TODO: stop propogation at gap_max (ptimisation)
    
  print("Propagated forward", len(real_idxs_to_f_prop))

  # Backward propagation then:

  b_prop_pts = []
  real_idxs_to_b_prop = {}
  b_prop_frames = [np.array([])] # init with last frame propagations (reversed array)

  #b_prop = np.ones((N, F, 2)) * np.Inf

  S = len(tracklets_by_start) - 1
  E = len(tracklets_by_end) - 1
  control_tracklets = set()
  for f in range(1, F):
    f = F - f - 1
    while E >= 0 and tracklets_by_end[ E ].end > f:
      t = tracklets_by_end[ E ]
      control_tracklets.add(t)
      E -= 1
    while S >= 0 and tracklets_by_start[ S ].start > f:
      t = tracklets_by_start[ S ]
      control_tracklets.discard(t)
      if t.start == f+1:
        real_idxs_to_b_prop[ t.id ] = len(b_prop_pts)
        b_prop_pts.append(t.points[ 0 ])
      S -= 1
    
    ctrl_p_s = [] # source control points
    ctrl_p_t = [] # target control points
    for t in control_tracklets:
      ctrl_p_s.append(t.point_at(f + 1 ))
      ctrl_p_t.append(t.point_at(f))
    
    if len(b_prop_pts) == 0:
      b_prop_pts_np = np.array([])
    else:
      b_prop_pts_np = TPS(np.array(ctrl_p_s), np.array(ctrl_p_t), np.array(b_prop_pts))
      b_prop_pts = b_prop_pts_np.tolist()
    b_prop_frames.append(b_prop_pts_np)
  b_prop_frames.reverse()

  print("Propagated backward", len(real_idxs_to_b_prop))

  #d = prop_f - prop_b
  #phi = np.minimum(phi, d)
  
  phi = np.ones((N, N)) * np.inf
  def update_phi(t1, t2):
    if t1.has_same_frames_with(t2): return
    if t2.start - t1.end > gap_max + 1  or t1.start - t2.end > gap_max + 1: return
    for f in range(F):
      p1 = t1.point_at(f)
      if p1 is None:
        if t1.end < f:
          p1 = f_prop_frames[f][real_idxs_to_f_prop[t1.id]]
        else:
          p1 = b_prop_frames[f][real_idxs_to_b_prop[t1.id]]
      p2 = t2.point_at(f)
      if p2 is None:
        if t2.end < f:
          p2 = f_prop_frames[f][real_idxs_to_f_prop[t2.id]]
        else:
          p2 = b_prop_frames[f][real_idxs_to_b_prop[t2.id]]
      d = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
      if d > d_max: continue
      phi[t1.id,t2.id] = phi[t2.id, t1.id] = min(phi[t1.id,t2.id], d)

  
  for i, t1 in enumerate(tracklets):
    for j in range(i):
      t2 = tracklets[j]
      update_phi(t1, t2)

    if i %100 == 0:
      print(f'{i+1} tracklets processed out of {N}, ')
      
  return phi