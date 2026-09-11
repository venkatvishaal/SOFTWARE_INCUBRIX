"""Shared pytest fixtures and configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that require GPU or long runtimes")
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")
