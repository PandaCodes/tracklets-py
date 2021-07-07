import xml.etree.ElementTree as ET

def tracklets_to_np(tracklets):
  N = len(tracklets)
  t_max = 0
  for t in tracklets:
      t_max = max(t_max, t.end)
  trs_np = np.ones((N, t_max + 1, 2)) * np.Inf
  for i, t in enumerate(tracklets):
    trs_np[i, t.start: t.end+1, : ] = np.array(t.points)
  return trs_np


def read_tracklets_from_xml(filename):
  tree = ET.parse(filename)
  root = tree.getroot()
  tracklets = []
  for child in root:
    if child.tag == 'trackgroup':
      root = child
      break
  for child in root:
    if child.tag == 'track':
      tracklets.append(Tracklet(
        int(child[0].attrib['t']), 
        [ (float(d.attrib['x']), float(d.attrib['y'])) for d in child if d.tag == 'detection' ],
        len(tracklets)
      ))
  return tracklets