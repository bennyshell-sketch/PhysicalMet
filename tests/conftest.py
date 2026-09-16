import warnings
from pathlib import Path

import pytest

from poraf_bt.config import SETTINGS
from poraf_bt.data_pipeline import build_dataset
from poraf_bt.generate_synthetic_data import GenerationParams, generate_workbook


@pytest.fixture(scope="session")
def workbook(tmp_path_factory) -> Path:
    """A null-effect synthetic workbook, generated once per test session."""
    out = tmp_path_factory.mktemp("data") / "synthetic.xlsx"
    path, _ = generate_workbook(GenerationParams(seed=7), output_path=out)
    return path


@pytest.fixture(scope="session")
def dataset(workbook):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_dataset(workbook)
