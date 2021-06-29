

# imports - standard imports
import subprocess as sp

# imports - module imports
from deeply.util.system import popen
from deeply.exception   import (
    DeeplyError,
    PopenError
)

# imports - test imports
import pytest

def test_deeply_error():
    with pytest.raises(DeeplyError):
        raise DeeplyError

def test_popen_error():
    with pytest.raises(PopenError):
        popen('python -c "from deeply.exceptions import PopenError; raise PopenError"')

    assert isinstance(
        PopenError(0, "echo foobar"),
        (DeeplyError, sp.CalledProcessError)
    )
    assert isinstance(DeeplyError(), Exception)