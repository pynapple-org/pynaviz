from contextlib import nullcontext as does_not_raise

import pytest

import pynaviz.synchronization_rules as sync_rules


class TestMatchPanOnXAxis:

    @pytest.mark.parametrize("update_type, expectation",
                             [
                                 ("pan", does_not_raise()),
                                 ("zoom", pytest.raises(ValueError, match="Update rule/event mismatch")),
                                 ("unknown", pytest.raises(ValueError, match="Update rule/event mismatch"))
                             ])
    def test_update_type(self, update_type, expectation, event_pan_update, camera_state):
        event_pan_update.update_type = update_type
        with expectation:
            sync_rules._match_pan_on_x_axis(event_pan_update, camera_state=camera_state)

    def test_update_output(self, event_pan_update, camera_state):
        event_pan_update.update_type = "pan"
        out = sync_rules._match_pan_on_x_axis(event_pan_update, camera_state=camera_state)
        assert isinstance(out, dict)
        assert tuple(out.keys()) == ("position",)


class TestMatchZoomOnXAxis:

    @pytest.mark.parametrize("update_type, expectation",
                             [
                                 ("pan", pytest.raises(ValueError, match="Update rule/event mismatch")),
                                 ("zoom", does_not_raise()),
                                 ("zoom_to_point", does_not_raise()),
                                 ("unknown", pytest.raises(ValueError, match="Update rule/event mismatch"))
                             ])
    def test_update_type(self, update_type, expectation, event_pan_update, camera_state):
        event_pan_update.update_type = update_type
        with expectation:
            sync_rules._match_zoom_on_x_axis(event_pan_update, camera_state=camera_state)

    def test_update_output(self, event_pan_update, camera_state):
        event_pan_update.update_type = "zoom"
        out = sync_rules._match_zoom_on_x_axis(event_pan_update, camera_state=camera_state)
        assert isinstance(out, dict)
        assert tuple(out.keys()) == ("position", "width")
