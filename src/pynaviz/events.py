"""Custom pygfx events."""

from typing import Optional

from pygfx import Event


class SyncEvent(Event):
    """
    Custom event to synchronize multiple controllers.

    Parameters
    ----------
    args
    controller_id
    update_type
    sync_extra_args
    kwargs
    """
    def __init__(
        self,
        *args,
        controller_id: Optional[int] = None,
        update_type: Optional[str] = "",
        sync_extra_args=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_id = controller_id
        self.update_type = update_type

        if sync_extra_args:
            self.args = sync_extra_args["args"]
            self.kwargs = sync_extra_args["kwargs"]

class SwitchEvent(Event):
    """
    Custom event to switch controllers. Useful when controller
    go from PanController to GetController for instance.

    Parameters
    ----------
    args
    controller_id
    update_type
    sync_extra_args
    kwargs
    """
    def __init__(
        self,
        *args,
        controller_id: Optional[int] = None,
        update_type: Optional[str] = "",
        new_controller: Optional[object] = None,
        sync_extra_args=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_id = controller_id
        self.update_type = update_type
        self.new_controller = new_controller

        if sync_extra_args:
            self.args = sync_extra_args["args"]
            self.kwargs = sync_extra_args["kwargs"]
