"""
Test script
"""

import numpy as np
import pynapple as nap
import pynaviz as viz

tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsd2 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsd3 = nap.Tsd(t=np.arange(1000), d=np.arange(1000))

viz.plot(*[tsd1, tsd2, tsd3])
