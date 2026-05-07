# Pynaviz Widget Examples

This guide shows how to generate example data using [Pynapple](https://github.com/pynapple-org/pynapple) 
and visualize it using the widgets provided by [`pynaviz`](https://github.com/pynapple-org/pynaviz).

## Prerequisites

To display Qt Widget, make sure PyQt6 is installed along with pynaviz 

```bash
pip install pynaviz[pyqt6]
```

## TsdWidget

```python
import numpy as np
import pynapple as nap
import pynaviz as viz

# Create a Tsd
tsd = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))

# Show it in the viewer
v = viz.TsdWidget(tsd)
v.show()
```

## TsdFrameWidget

```python
import numpy as np
import pynapple as nap
import pynaviz as viz

# Create a TsdFrame with two dimensions (cos and sin of a ramp)
tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((
        np.cos(2 * np.pi * np.arange(0, 100, 0.1)),
        np.sin(2 * np.pi * np.arange(0, 100, 0.1))
    )).T,
    metadata={"label": np.random.randn(2)}
)

# Show it in the viewer
v = viz.TsdFrameWidget(tsdframe)
v.show()
```

## TsdTensorWidget

```python
import numpy as np
import pynapple as nap
import pynaviz as viz

# Create a TsdTensor (e.g. 10x10 channels over 1000 time steps)
tsdtensor = nap.TsdTensor(
    t=np.arange(1000),
    d=np.random.randn(1000, 10, 10)
)

# Show it in the viewer
v = viz.TsdTensorWidget(tsdtensor)
v.show()
```

## TsGroupWidget

```python
import numpy as np
import pynapple as nap
import pynaviz as viz

# Simulate 10 spike trains with varying densities
tsg = nap.TsGroup({
    i: nap.Ts(t=np.sort(np.random.uniform(0, 1000, 100 * (i + 1))))
    for i in range(10)
}, metadata={"label": np.random.randn(10)})

# Display using TsGroupWidget
v = viz.TsGroupWidget(tsg)
v.show()
```

