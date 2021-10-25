# tracklets-py

Elastic motion correction and concatenation (EMC2) algorithm implementation. [Preprint is here.](https://www.biorxiv.org/content/10.1101/2020.06.22.165696v1.abstract) 
Along with a set of tools for a work with tracklets, which are essentially sequences of points in timeframes.


*Tracklets.py* - the class for a tracklet abstraction
*plot.py* - set of methods to plot tracklets
*tps.py* - Thin plate spline transformation implementation
*emc2.py* - Elastic motion correction and concatenation (EMC2) algorithm implementation

Usage:

```python
import tracklets

trls = tracklets.read_from_xml("data.xml")
phi = tracklets.emc2.make_phi(trs, gap_max, d_max)
```
The `phi` is the association cost matrix.
To have the final concatenated pairs one should solve an assignment problem by minimising the global distance. This can be done, for example, with the Jonker-Volgenant algorithm implemented by [this](https://github.com/src-d/lapjv) python library.

```python
from lapjv import lapjv

row_ind, col_ind, _ = lapjv(phi)

concat_pairs = list(set([
  *[ (i,j) for i, j in enumerate(row_ind) if i < j ],
  *[ (j,i) for i, j in enumerate(row_ind) if j >= 0 and j < i ],
  *[ (i,j) for i, j in enumerate(col_ind) if i < j ],
  *[ (j,i) for i, j in enumerate(col_ind) if j >= 0 and j < i ],
]))
# concat_pairs.sort(key=lambda x:x[0])

```

TODO: 
- Algorithm automated tests (+ tests description)
- Concatenation based on TPS values
- Optimisation. Yes, this is python, but it takes too long comparing to lapjv, which is the most heavy part in Java implementation. Something seems to be wrong here.
- XML format description + translations to other formats




*Repository developement to be continued...*