import pynaviz.events as events


def test_sync_event_initialization():
    ev = events.SyncEvent("mytype", update_type="up", controller_id=1, sync_extra_args={"args": (1, 2, 3), "kwargs": dict(one=1, two=2)})
    assert ev.type == "mytype"
    assert ev.update_type == "up"
    assert ev.controller_id == 1
    assert ev.args == (1, 2, 3)
    assert tuple(ev.kwargs.values()) == (1, 2)
    assert tuple(ev.kwargs.keys()) == ("one", "two")
