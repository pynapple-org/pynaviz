from typing import *
from fastplotlib import ImageGraphic, LinearSelector
from ipywidgets import IntSlider, FloatSlider
from pynapple import TsdFrame

from fastplotlib.graphics._features import FeatureEvent


class TimeStoreComponent:
    @property
    def subscriber(self) -> Union[ImageGraphic, IntSlider, FloatSlider, LinearSelector]:
        return self._subscriber

    @property
    def data(self) -> Union[TsdFrame, None]:
        return self._data

    @property
    def multiplier(self) -> Union[int, float, None]:
        return self._multiplier

    def __init__(self, subscriber, data=None, multiplier=None):
        """A TimeStore component of the time store."""
        if multiplier is None:
            multiplier = 1

        self._multiplier = multiplier

        self._subscriber = subscriber

        # must have data if ImageGraphic
        if isinstance(self.subscriber, ImageGraphic):
            if not isinstance(data, TsdFrame):
                raise ValueError("If passing in `ImageGraphic` must provide associated `TsdFrame` to update data with.")
            self._data = data


class TimeStore:
    @property
    def time(self):
        """Current t value that items in the store are set at."""
        return self._time

    @time.setter
    def time(self, value: int | float):
        """Set the current time."""
        self._time = value

    @property
    def store(self) -> List[TimeStoreComponent]:
        """Returns the items in the store."""
        return self._store

    def __init__(self):
        """
        TimeStore for synchronizes and updating components of a plot (i.e. Ipywidgets.IntSlider,
        fastplotlib.LinearSelector, or fastplotlob.ImageGraphic).

        NOTE: If passing a `fastplotlib.ImageGraphic`, it is understood that there should be an associated
        `pynapple.TsdFrame` given.
        """
        # initialize store
        self._store = list()
        # by default, time is zero
        self._time = 0

    def subscribe(self,
                  subscriber: Union[ImageGraphic, LinearSelector, IntSlider, FloatSlider],
                  data: TsdFrame = None,
                  multiplier: int | float = None) -> None:
        """
        Method for adding a subscriber to the store to be synchronized.

        Parameters
        ----------
        subscriber: fastplotlib.ImageGraphic, fastplotlib.LinearSelector, ipywidgets.IntSlider, or ipywidgets.FloatSlider
            ipywidget or fastplotlib object to be synchronized
        data: pynapple.TsdFrame, optional
            If subscriber is a fastplotlib.ImageGraphic, must have an associating pynapple.TsdFrame to update data with.
        multiplier: int | float, optional
            Scale the current time to reflect differing timescale.
        """
        # create a TimeStoreComponent
        component = TimeStoreComponent(subscriber, data, multiplier)

        # add component to the store
        self._store.append(component)

        # add event handler to component.subscriber to call update_store
        if isinstance(component.subscriber, (IntSlider, FloatSlider)):
            component.subscriber.observe(self._update_store, "value")
        if isinstance(component.subscriber, LinearSelector):
            component.subscriber.add_event_handler(self._update_store, "selection")

    def _update_store(self, ev):
        """Called when event occurs and store needs to be updated."""
        # parse event to see if it originated from ipywidget or selector
        if isinstance(ev, FeatureEvent):
            # check for multiplier to adjust time
            for component in self.store:
                if isinstance(component.subscriber, LinearSelector):
                    if ev.graphic == component.subscriber:
                        self.time = ev.info["value"] / component.multiplier
        else:
            self.time = ev["new"]

        for component in self.store:
            # update ImageGraphic data no matter what
            if isinstance(component.subscriber, ImageGraphic):
                ImageGraphic.data = component.data.get(self.time)
            elif isinstance(component.subscriber, LinearSelector):
                # only update if different
                if component.subscriber.selection != self.time * component.multiplier:
                    component.subscriber.selection = self.time * component.multiplier
            else:
                # only update if different
                if component.subscriber.value != self.time:
                    component.subscriber.value = self.time
