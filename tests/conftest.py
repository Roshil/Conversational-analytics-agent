"""Shared fixtures. Full-size generation runs once per test session (~15 s)."""
import pytest

from analytics_agent.data_generation.corruption import corrupt
from analytics_agent.data_generation.generator import GenerationConfig, generate_clean


@pytest.fixture(scope="session")
def cfg():
    return GenerationConfig()


@pytest.fixture(scope="session")
def clean(cfg):
    return generate_clean(cfg)


@pytest.fixture(scope="session")
def corrupted_bundle(clean):
    return corrupt(clean, seed=777)


@pytest.fixture(scope="session")
def corrupted(corrupted_bundle):
    return corrupted_bundle[0]


@pytest.fixture(scope="session")
def defects(corrupted_bundle):
    return corrupted_bundle[1]
