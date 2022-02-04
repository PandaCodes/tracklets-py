import xml.etree.ElementTree as ET

from tracklets import Tracklet


def read_xml(filename):
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


