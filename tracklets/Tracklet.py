from __future__ import annotations # for python < 3.10
from typing import Optional

Point = tuple[float, float]

class Tracklet:
  # start: first frame tracklet occured
  # points: list of points starting from that frame
  # tr_id: tracklet id for use inside data structures
  def __init__(self, start: int, points: list[Point], tr_id = None):
    if len(points) == 0: raise Error("Tracklet: no points passed")
    if start < 0: raise Error("Start should be a non-negative value")
    self.start = start
    self.points = points
    self.end = start + len(points) - 1
    self.id = tr_id
  
  def point_at(self, frame: int) -> Optional[Point]:
    p_index = frame - self.start
    if p_index < 0 or p_index >= len(self.points):
      return None
    return self.points[ p_index ]
  
  def has_same_frames_with(self, t: Tracklet) -> bool:
    return (self.end >= t.start and self.end <= t.end) or (t.end >= self.start and t.end <= self.end)

  def get_3d_coordinates(self) -> list[tuple[float, float, int]]:
    return [(p[0], p[1], self.start + i) for i, p in enumerate(self.points)]

