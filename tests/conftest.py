import pytest

from flotilla import Flotilla
from flotilla.backends import BwrapBackend

HAVE_BWRAP = BwrapBackend.available()[0]
BACKENDS = ["local"] + (["bwrap"] if HAVE_BWRAP else [])


@pytest.fixture(params=BACKENDS)
def fl(request, tmp_path):
    return Flotilla(backend=request.param, state_dir=str(tmp_path), max_concurrency=4)


@pytest.fixture
def iso(tmp_path):
    if not HAVE_BWRAP:
        pytest.skip("bwrap not available")
    return Flotilla(backend="bwrap", state_dir=str(tmp_path), max_concurrency=4)
