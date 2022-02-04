from __future__ import annotations # for python < 3.10
from typing import Optional

import numpy as np

Point = tuple[float, float]

class Tracklet:
  # start: first frame tracklet occured
  # points: list of points starting from that frame
  # id: tracklet id for use inside data structures
  def __init__(self, start: int, points: list[Point], id = None):
    if len(points) == 0: raise Error("Tracklet: no points passed")
    if start < 0: raise Error("Start should be a non-negative integer")
    self.start = start
    self.points = points
    self.end = start + len(points)  #TODO: inambiguious end/last_frame
    self.id = id
  
  @property
  def length(self):
    return len(self.points)

  @property
  def first_frame(self):
    return self.start

  @property
  def last_frame(self):
    return self.end - 1

  def point_at(self, frame: int) -> Optional[Point]:
    p_index = frame - self.start
    if p_index < 0 or p_index >= len(self.points):
      return None
    return self.points[ p_index ]
  
  def has_same_frames_with(self, t: Tracklet) -> bool:
    return (self.end >= t.start and self.end <= t.end) or (t.end >= self.start and t.end <= self.end)

  def get_3d_coordinates(self) -> list[tuple[float, float, int]]:
    return [(p[0], p[1], self.start + i) for i, p in enumerate(self.points)]


def to_np(tracklets: [Tracklet]):
  # t000 = time.time() 
  N = len(tracklets)
  # Figuring out frames count:
  F = 0 
  for tr in tracklets:
    F = max(F, tr.end)

  # Tracklets to numpy array
  points = np.full((F, N, 2), np.nan)
  for i, t in enumerate(tracklets):
    p_frame_idx = range(t.start, t.end)
    points[p_frame_idx, i, : ] = t.points

  return points
