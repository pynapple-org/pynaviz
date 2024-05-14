"""
	The base class that controls everything.
"""
from .controller import Controller

class Base():

	def __init__(self, *args):
		self._args = args
		self._controllers = {}
		self._plots = {}

		for i, obj in enumerate(args):
			
			# instantiate fastplotlib
			self._plots[i] = typepynapple(obj)

			camera = self._plots[i].camera
			renderer = self._plots[i].renderer

			# attach the controller
			self._controllers[tsd_id] = Controller(tsd_id, camera, renderer)