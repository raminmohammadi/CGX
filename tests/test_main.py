import pytest

@pytest.fixture
def temp_path(tmp_path):
    return tmp_path