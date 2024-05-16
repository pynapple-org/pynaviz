"""Custom pygfx events."""
from pygfx import Event
from typing import Optional


class SyncEvent(Event):
    def __init__(
            self,
            *args,
            controller_id: Optional[int] = None,
            update_type: Optional[str] = "",
            data,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controller_id = controller_id
        self.update_type = update_type
        self.args = data["args"]
        self.kwargs = data["kwargs"]
