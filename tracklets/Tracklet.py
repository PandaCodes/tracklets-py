class Tracklet:
  def __init__(self, start, points, index):
    if len(points) == 0: raise Error("No points in tracklet initialization")
    self.start = start
    self.points = points
    self.end = start + len(points) - 1
    self.index = index
  
  def point_at(self, frame):
    p_index = frame - self.start
    if p_index < 0 or p_index >= len(self.points):
      return None
    return self.points[ p_index ]
  
  def has_same_frames_with(self, t):
    return (self.end >= t.start and self.end <= t.end) or (t.end >= self.start and t.end <= self.end)

  def get_3d_coordinates(self):
    return [(p[0], p[1], self.start + i) for i, p in enumerate(self.points)]

