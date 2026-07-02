"""Shared pytest fixtures and import-path setup."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                       # constants, services, main
sys.path.insert(0, os.path.join(ROOT, "src"))  # features


@pytest.fixture(scope="session")
def data_service():
    from services.f1_data_service import F1DataService
    return F1DataService()


@pytest.fixture(scope="session")
def prediction_service():
    from services.prediction_service import PredictionService
    return PredictionService()


@pytest.fixture(scope="session")
def a_race(data_service):
    """A real race that is guaranteed to exist in the dataset."""
    races = data_service.races_by_season()
    season = max(races)                 # newest season
    circuit = races[season][0]          # its first round
    return season, circuit, data_service.get_race(season, circuit)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)
