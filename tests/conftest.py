"""
Shared pytest fixtures.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def ensemble_config_path():
    return ROOT / "models" / "ensemble_final" / "config.json"


@pytest.fixture(scope="session")
def label_map_path():
    return ROOT / "models" / "label_map.json"


@pytest.fixture(scope="session")
def has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
