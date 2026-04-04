import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "carla: requires live CARLA server")
