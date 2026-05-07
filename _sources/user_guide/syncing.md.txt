# Time Series Synchronization

Synchronization occurs through the `ControllerGroup` object that deals with time sync.

## Simple example

```python
import numpy as np
import pynapple as nap
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

# Data
tsd = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

# Creation of visuals
viz1 = viz.PlotTsd(tsd)
viz2 = viz.PlotTsdTensor(tsdtensor)

# Controller group
ctrl_group = ControllerGroup([viz1, viz2])

# One visual need to run
viz1.show() 
```

## With Qt

Qt allows to embed pygfx canvas in a single windows.

```python
import numpy as np
import pynapple as nap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QWidget
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup

# Data
tsd = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

# Qt Application
app = QApplication([])
window = QWidget()
window.setMinimumSize(1500, 800)
layout = QHBoxLayout()

# Creation of visuals
viz1 = viz.TsdWidget(tsd)
viz2 = viz.TsdTensorWidget(tsdtensor)

# Controller group
ctrl_group = ControllerGroup()
ctrl_group.add(viz1, 0)
ctrl_group.add(viz2, 1)

# Adding the widgets to the layout
layout.addWidget(viz1)
layout.addWidget(viz2)
window.setLayout(layout)
window.show()

if __name__ == "__main__":
    app.exit(app.exec())
```