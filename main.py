"""
	Test script
"""

import numpy as np
import pynapple as nap
import pynaviz as viz

tsd1 = nap.Tsd(t=np.arange(100), d = np.cos(np.arange(100)))
tsd2 = nap.Tsd(t=np.arange(100), d = np.sin(np.arange(100)))
tsd3 = nap.Tsd(t=np.arange(100), d = np.arange(100))

viz.plot(*[tsd1,tsd2,tsd3])

