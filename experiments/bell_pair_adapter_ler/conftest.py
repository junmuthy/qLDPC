"""Archive checks are opt-in; ordinary tests never need saved simulation output."""

import pytest

collect_ignore = ["outputs"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--adapter-archives", action="store_true", help="Also validate local adapter-LER archives"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "adapter_archive: requires local archived adapter outputs")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--adapter-archives"):
        for item in items:
            if item.get_closest_marker("adapter_archive"):
                item.add_marker(
                    pytest.mark.skip(reason="requires --adapter-archives and local data")
                )
