"""Custom pygfx events."""

from typing import Optional

from pygfx import Event


class SyncEvent(Event):
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
