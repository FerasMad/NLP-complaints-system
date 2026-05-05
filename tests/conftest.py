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
def schema_version(ensemble_config_path):
    """Read schema_version from ensemble config; default to '8class' for back-compat.

    Production uses '8class' (the deployed model). The v5 ambience experiment
    uses '9class_ambience'. Tests that touch the label map should branch on
    this fixture so both schemas can coexist in the repo.
    """
    import json
    if not ensemble_config_path.exists():
        return "8class"
    try:
        with open(ensemble_config_path, encoding="utf-8") as f:
            return json.load(f).get("schema_version", "8class")
    except Exception:
        return "8class"


@pytest.fixture(scope="session")
def has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
