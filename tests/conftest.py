"""Shared fixtures: one seeded hotel group, its survey, and the raked result."""

from __future__ import annotations

import pytest

from surveyrake.pipeline import weight_quarters
from surveyrake.synth import draw_responses, generate_population

POP_SEED = 2025
RESPONSE_SEED = 2


@pytest.fixture(scope="session")
def population():
    return generate_population(POP_SEED)


@pytest.fixture(scope="session")
def population_table(population):
    return population.margin_table()


@pytest.fixture(scope="session")
def responses(population):
    return draw_responses(population, RESPONSE_SEED)


@pytest.fixture(scope="session")
def weighted(responses, population_table):
    frame, _ = weight_quarters(responses, population_table)
    return frame


@pytest.fixture(scope="session")
def rake_reports(responses, population_table):
    _, reports = weight_quarters(responses, population_table)
    return reports
