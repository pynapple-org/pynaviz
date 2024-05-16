import pytest
import pynaviz.synchronization_rules as sync_rules
from contextlib import nullcontext as does_not_raise


class TestMatchPanOnXAxis:

    @pytest.mark.parametrize("update_type, expectation",
                             [
                                 ("pan", does_not_raise()),
                                 ("zoom", pytest.raises(ValueError, match="Update type for")),
                                 ("unknown", pytest.raises(ValueError, match="Update type for"))
                             ])
    def test_update_type(self, update_type, expectation, event_pan_update):
        event_pan_update.update_type = update_type
        with expectation:
            sync_rules._match_pan_on_x_axis(event_pan_update, camera_state=event_pan_update.kwargs["cam_state"])