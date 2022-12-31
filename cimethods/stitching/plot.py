from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

x_min = 1e50
x_max = 0
y_min = 1e50
y_max = 0
t_max = 0
for t in tracklets:
    t_max = max(t_max, t.end)
    for p in t.points:
        x_min = min(x_min, p[0])
        x_max = max(x_max, p[0])
        y_min = min(y_min, p[1])
        y_max = max(y_max, p[1])

print(t_max, x_min, x_max, y_min, y_max)

#df = pd.DataFrame(data = points, columns = ["x", "y", "t", "c"])
def plot_lines(points):
  df = pd.DataFrame(data = points, columns = ["x", "y", "t", "c"])
  fig = px.line_3d(df, x="x", y="y", z="t", color="c")
  fig.update_layout(
      scene = dict(
          xaxis = dict(nticks=4, range=[x_min,x_max],),
          yaxis = dict(nticks=4, range=[y_min,y_max],),
          zaxis = dict(nticks=4, range=[0,t_max],),),
      width=700,
  )
  iplot(fig)


def plot_tracklets(tracklets):
  tr_pts = None
  for i, tr in enumerate(tracklets):
    l = len(tr.points)
    t = np.linspace(tr.start, tr.end, l)
    c = np.ones(l) * i
    pts = np.array(tr.points)
    pts = np.vstack([pts[:,0], pts[:,1], t, c]).T
    if tr_pts is None:
      tr_pts = pts
    else:
      tr_pts = np.concatenate((tr_pts, pts))
  plot_lines(tr_pts)


  
# import matplotlib.pyplot as plt
# import random

# def plot_tracklets(tracklets, alter_linestyle = False, alpha = 0.5, ax = None):
#   if ax == None:
#     fig = plt.figure(figsize=(12, 9), dpi=80)
#     ax = plt.axes(projection='3d')
#   for i, t in enumerate(tracklets):
#       points = t.get_3d_coordinates()
#       xline = [p[0] for p in points]
#       yline = [p[1] for p in points]
#       tline = [p[2] for p in points]

#       color = (random.random(), random.random(), random.random())
#       ls = ['-','--','-.',':'][i%4] if alter_linestyle else '-'
#       ax.plot3D(xline, yline, tline, c=color, linestyle=ls, alpha=alpha)
#   return ax