# imports - module imports
from deeply.exception import (
    DeeplyError
)

# imports - test imports
import pytest

def test_deeply_error():
    with pytest.raises(DeeplyError):
        raise DeeplyError
