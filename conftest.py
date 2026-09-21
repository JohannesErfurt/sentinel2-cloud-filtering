"""Root conftest: command-line options and markers.

``pytest_addoption`` is only honoured in the rootdir conftest, so it lives here
while the fixtures stay in ``tests/conftest.py``.
"""
import os


def pytest_addoption(parser):
    parser.addoption(
        "--safe-dir",
        action="store",
        default=os.environ.get("S2_SAFE_DIR"),
        help="path to a Sentinel-2 L1C .SAFE folder; enables the tests marked needs_product",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_product: requires a real .SAFE folder via --safe-dir"
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    if config.getoption("--safe-dir"):
        return
    skip = pytest.mark.skip(reason="needs --safe-dir (a real .SAFE product)")
    for item in items:
        if "needs_product" in item.keywords:
            item.add_marker(skip)
