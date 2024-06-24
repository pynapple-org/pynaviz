from typing import *
from fastplotlib import ImageGraphic, LinearSelector, ScatterGraphic
from ipywidgets import IntSlider, FloatSlider
from pynapple import TsdFrame, TsdTensor

from fastplotlib.graphics._features import FeatureEvent

MARGIN: float = 1


# TODO: need to make a method for automatic MARGIN setting based on the data


class TimeStoreComponent:
    @property
    def subscriber(self) -> ImageGraphic | IntSlider | FloatSlider | LinearSelector:
        return self._subscriber

    @property
    def data(self) -> TsdFrame | TsdTensor | None:
        return self._data

    @property
    def multiplier(self) -> int | float | None:
        return self._multiplier

    @property
    def data_filter(self) -> callable:
        return self._data_filter

    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """A TimeStore component of the time store."""
        if multiplier is None:
            multiplier = 1

        self._multiplier = multiplier

        self._subscriber = subscriber

        # must have data if ImageGraphic
        if isinstance(self.subscriber, (ImageGraphic, ScatterGraphic)):
            if not isinstance(data, (TsdFrame, TsdTensor)):
                raise ValueError("If passing in `ImageGraphic` must provide associated `TsdFrame` to update data with.")
            self._data = data

        self._data_filter = data_filter


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
                  subscriber: ImageGraphic | LinearSelector | ScatterGraphic | IntSlider | FloatSlider,
                  data: TsdFrame | TsdTensor = None,
                  data_filter: callable = None,
                  multiplier: int | float = None) -> None:
        """
        Method for adding a subscriber to the store to be synchronized.

        Parameters
        ----------
        subscriber: fastplotlib.ImageGraphic, fastplotlib.LinearSelector, ipywidgets.IntSlider, or ipywidgets.FloatSlider
            ipywidget or fastplotlib object to be synchronized
        data: pynapple.TsdFrame, optional
            If subscriber is a fastplotlib.ImageGraphic, must have an associating pynapple.TsdFrame to update data with.
        data_filter: callable, optional
            Function to apply to data before updating. Must return data in the same shape as input.
        multiplier: int | float, optional
            Scale the current time to reflect differing timescale.
        """
        # create a TimeStoreComponent
        component = TimeStoreComponent(subscriber=subscriber,
                                       data=data,
                                       data_filter=data_filter,
                                       multiplier=multiplier)

        # add component to the store
        self._store.append(component)

        # add event handler to component.subscriber to call update_store
        if isinstance(component.subscriber, (IntSlider, FloatSlider)):
            component.subscriber.observe(self._update_store, "value")
        if isinstance(component.subscriber, LinearSelector):
            component.subscriber.add_event_handler(self._update_store, "selection")

    def unsubscribe(self, subscriber: ImageGraphic | LinearSelector | IntSlider | FloatSlider):
        """Remove a subscriber from the store."""
        for component in self.store:
            if component.subscriber == subscriber:
                #  remove the component from the store
                self.store.remove(component)
                # remove event handler
                if isinstance(component, (IntSlider, FloatSlider)):
                    component.unobserve(self._update_store)
                if isinstance(component, LinearSelector):
                    component.subscriber.remove_event_handler(self._update_store, "selection")

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
            if isinstance(component.subscriber, ScatterGraphic):
                component.subscriber.data = component.data.get(self.time)
            elif isinstance(component.subscriber, ImageGraphic):
                if component.data_filter is None:
                    new_data = component.data.get(self.time)
                else:
                    new_data = component.data_filter(component.data.get(self.time))
                if new_data.shape != component.subscriber.data.value.shape:
                    raise ValueError(f"data filter function: {component.data_filter} must return data in the same shape"
                                     f"as the current data")
                component.subscriber.data = new_data
            elif isinstance(component.subscriber, LinearSelector):
                # only update if different
                if abs(component.subscriber.selection - (self.time * component.multiplier)) > MARGIN:
                    component.subscriber.selection = self.time * component.multiplier
            else:
                # only update if different
                if abs(component.subscriber.value - self.time) > MARGIN:
                    component.subscriber.value = self.time