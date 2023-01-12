## Stitching


A set of tools for a work with tracklets, which are essentially sequences of points in timeframes.
Originally build around Elastic motion correction and concatenation (EMC2) algorithm. (Please, find the [original paper here](https://doi.org/10.1371/journal.pcbi.1009432))


*Tracklets.py* - the class for a tracklet abstraction + conversion of the tracklets list to numpy representation
*tps.py* - Thin plate spline (TPS) transformation implementation
*emc2.py* - Elastic motion correction and concatenation (EMC2) algorithm implementation
*plot.py* - set of methods to plot tracklets


### Tracklets
Library uses 2 types of tracklet data representation. The first one is a `Tracklet` class, that incapsulates several relevant properties and methods.
```python
# python 3 required
import tracklets

trl = Tracklet(start=20, points=[(21, 35), (22, 34.5), ... ])
```
The list of tracklets can be read from the xml file as well:
```py
trls = tracklets.read_xml("data.xml")
```

Another representation of the list of tracklets is a numpy array of the shape (F, N, 2), where F-dimention is for frames, N-dimention is tracklets, 2-dimention is _x_ and _y_ coordinates. If there is no point for a tracklet `n` at a frame `f` presented then the `np_trls[f,n,:]` is set to `[ np.nan, np.nan]`.
It can be obtained from the list of tracklets by a `to_np` function:
```py
np_trls = tracklets.to_np(trls)
```

### EMC2

Usage:

```python
phi, fwd_prop, bwd_prop  = tracklets.emc2.make_phi(trls, gap_max=200, d_max=10)
```
The `phi` is the association cost matrix. To have the final concatenated pairs one should solve an assignment problem by minimising the global distance. This can be done, for example, with the Jonker-Volgenant algorithm implemented in [Lap](https://github.com/gatagat/lap) library:

```python
import lap

_, x, _ = lap.lapjv(phi)

```


`fwd_prop` and `bwd_prop` are tracklets propagations estimated with TPS. Both are in the numpy representation. They contain propagations only, without points of the original tracklets.

It is possible to use `tracklets.calc_propagation` to obtain these matrixes alone:
```py
fwd_prop = tracklets.calc_propagation(np_trls, max_len=200)
bwd_prop = tracklets.calc_propagation(np_trls, max_len=200, direction="backward")

```
`max_len` is the maximum length (in number of frames) to propagate on. Default is `np.inf`.



## Tests:
To execute tests, run:
```sh
pytest
```
There is a test marked `visual` - the one to evaluate visually if TPS transform gives a reasonable result.

---

TODO: 
- Algorithm tests
- XML format description + i/o 
- Translations to other formats
- Optimisation.
