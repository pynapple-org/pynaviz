"""
	The high level function to call.
	Ideally something like :

	>>> import pynaviz as viz
	>>> viz.plot(tsd)
"""
from .base import Base

def plot(*args):

	return Base(*args)
		