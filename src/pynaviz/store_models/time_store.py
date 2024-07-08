from typing import *
from fastplotlib import ImageGraphic, LinearSelector, ScatterGraphic
from ipywidgets import IntSlider, FloatSlider
from pynapple import TsdFrame, TsdTensor
import numpy as np

from fastplotlib.graphics._features import FeatureEvent

MARGIN: float = 1


class TimeStoreComponent:
    @property
    def subscriber(self) -> ImageGraphic | IntSlider | FloatSlider | LinearSelector | ScatterGraphic:
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
        """Base class for a component of the TimeStore."""
        if multiplier is None:
            multiplier = 1

        self._multiplier = multiplier

        self._subscriber = subscriber

        self._data = data

        self._data_filter = data_filter

    def update(self, time: int | float):
        raise NotImplementedError("Must be implemented in subclass TimeStoreComponent")


class TimeStoreImageComponent(TimeStoreComponent):
    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """ImageGraphic TimeStore component."""
        if not isinstance(data, (TsdFrame, TsdTensor)):
            raise ValueError("If passing in `ImageGraphic` must provide associated `TsdFrame` to update data with.")

        super().__init__(
            subscriber=subscriber,
            data=data,
            data_filter=data_filter,
            multiplier=multiplier
        )

    def update(self, time: float | int):
        """Update the ImageGraphic data."""
        if self.data_filter is None:
            new_data = self.data.get(time)
        else:
            new_data = self.data_filter(self.data.get(time))
        if new_data.shape != self.subscriber.data.value.shape:
            raise ValueError(f"data filter function: {self.data_filter} must return data in the same shape"
                             f"as the current data")
        self.subscriber.data = new_data


# TODO: will eventually get replaced with BallSelector and then can just be integrated into TimeStoreSelectorComponent
class TimeStoreScatterComponent(TimeStoreComponent):
    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """TimeStoreComponent for ScatterGraphic."""
        if not isinstance(data, TsdTensor):
            raise ValueError("If passing in `ScatterGraphic` must provide associated `TsdTensor` to update data with.")

        super().__init__(
            subscriber=subscriber,
            data=data,
            data_filter=data_filter,
            multiplier=multiplier
        )

    def update(self, time: float | int):
        """Update the scatter data."""
        self.subscriber.data = self.data.get(time)


class TimeStoreSelectorComponent(TimeStoreComponent):
    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """TimeStoreComponent for LinearSelector."""

        super().__init__(
            subscriber=subscriber,
            data=data,
            data_filter=data_filter,
            multiplier=multiplier
        )

    def update(self, time: int | float):
        """Update the selection."""
        if self.subscriber.selection == time * self.multiplier:
            return
        self.subscriber.selection = time * self.multiplier


class TimeStoreWidgetComponent(TimeStoreComponent):
    def __init__(self, subscriber, data=None, data_filter=None, multiplier=None):
        """TimeStoreComponent for IntSlider or FloatSlider."""
        super().__init__(
            subscriber=subscriber,
            data=data,
            data_filter=data_filter,
            multiplier=multiplier
        )

    def update(self, time: int | float):
        """Update slider value."""
        if self.subscriber.value == time:
            return
        self.subscriber.value = time


class TimeStore:
    @property
    def time(self):
        """Current t value that items in the store are set at."""
        return self._time

    @time.setter
    def time(self, value: int | float):
        """Set the current time."""

        if value == self._time:
            return

        self._time = value
        self._update_store(
            ev={
                "new": value
            }
        )

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

        # parse the type of subscriber passed and instantiate the appropriate TimeStoreComponent subclass
        if isinstance(subscriber, ImageGraphic):
            component = TimeStoreImageComponent(subscriber, data, data_filter, multiplier)
        elif isinstance(subscriber, LinearSelector):
            component = TimeStoreSelectorComponent(subscriber, data, data_filter, multiplier)
            # add event handler for updating the store
            component.subscriber.add_event_handler(self._update_store, "selection")
        elif isinstance(subscriber, ScatterGraphic):
            component = TimeStoreScatterComponent(subscriber, data, data_filter, multiplier)
        else:
            component = TimeStoreWidgetComponent(subscriber, data, data_filter, multiplier)
            # add event handler for updating the store
            component.subscriber.observe(self._update_store, "value")

        # add component to the store
        self._store.append(component)

    def unsubscribe(self, subscriber: ImageGraphic | LinearSelector | IntSlider | FloatSlider):
        """Remove a subscriber from the store."""
        for component in self.store:
            if component.subscriber == subscriber:
                #  remove the component from the store
                self.store.remove(component)
                # remove event handler
                if isinstance(component.subscriber, (IntSlider, FloatSlider)):
                    component.subscriber.unobserve(self._update_store)
                if isinstance(component.subscriber, LinearSelector):
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
            component.update(self.time)
