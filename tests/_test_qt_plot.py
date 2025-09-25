import pytest

import pynaviz as viz


@pytest.fixture
def data(request):
    return request.getfixturevalue(request.param)

# @pytest.fixture(scope="session")
# def app():
#     app = QApplication.instance()
#     if app is None:
#         app = QApplication(sys.argv)
#     return app


@pytest.mark.parametrize(
    "func, data",
    [
        ("TsdWidget", "dummy_tsd"),
        ("TsdFrameWidget", "dummy_tsdframe"),
        # # ("TsdTensorWidget", "dummy_tsdtensor"),
        ("TsGroupWidget", "dummy_tsgroup"),
        ("IntervalSetWidget", "dummy_intervalset"),
    ], indirect=["data"]
)
def test_myapp_starts(func, data):
    getattr(viz, func)(data)
    # v.render()
    # v.show()
    # assert widget.isVisible()
    # assert widget.windowTitle() == "Test App"
