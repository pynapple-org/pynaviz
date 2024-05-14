"""
	Test script
"""

import numpy as np
import pynapple as nap
import pynaviz as viz

tsd = nap.Tsd(t=np.arange(100), d = np.cos(np.arange(100)))

viz.plot(tsd)

