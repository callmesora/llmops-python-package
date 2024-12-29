"""Configuration for the tests."""

# %% IMPORTS

import os
import typing as T

import omegaconf
import pytest
from _pytest import logging as pl

from llmops_project.io import services

# %% CONFIGS

LIMIT = 1500
N_SPLITS = 3
TEST_SIZE = 24 * 7  # 1 week

# %% FIXTURES

# %% - Paths


@pytest.fixture(scope="session")
def tests_path() -> str:
    """Return the path of the tests folder."""
    file = os.path.abspath(__file__)
    parent = os.path.dirname(file)
    return parent


@pytest.fixture(scope="session")
def data_path(tests_path: str) -> str:
    """Return the path of the data folder."""
    return os.path.join(tests_path, "data")


@pytest.fixture(scope="session")
def confs_path(tests_path: str) -> str:
    """Return the path of the confs folder."""
    return os.path.join(tests_path, "confs")


@pytest.fixture(scope="session")
def inputs_path(data_path: str) -> str:
    """Return the path of the inputs dataset."""
    return os.path.join(data_path, "inputs_sample.parquet")


@pytest.fixture(scope="session")
def targets_path(data_path: str) -> str:
    """Return the path of the targets dataset."""
    return os.path.join(data_path, "targets_sample.parquet")


@pytest.fixture(scope="session")
def outputs_path(data_path: str) -> str:
    """Return the path of the outputs dataset."""
    return os.path.join(data_path, "outputs_sample.parquet")


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory) -> str:
    """Create a session-scoped temporary directory."""
    return tmp_path_factory.mktemp("session_tmp")


@pytest.fixture(scope="session")
def tmp_outputs_path(session_tmp_path: str) -> str:
    """Return a session-scoped tmp path for the outputs dataset."""
    return os.path.join(session_tmp_path, "outputs.parquet")


@pytest.fixture(scope="session")
def tmp_models_explanations_path(session_tmp_path: str) -> str:
    """Return a session-scoped tmp path for the model explanations dataset."""
    return os.path.join(session_tmp_path, "models_explanations.parquet")


@pytest.fixture(scope="session")
def tmp_samples_explanations_path(session_tmp_path: str) -> str:
    """Return a session-scoped tmp path for the samples explanations dataset."""
    return os.path.join(session_tmp_path, "samples_explanations.parquet")


# %% - Configs


@pytest.fixture(scope="session")
def extra_config() -> str:
    """Extra config for scripts."""
    # use OmegaConf resolver: ${tmp_path:}
    config = """
    {
        "job": {
            "alerts_service": {
                "enable": false,
            },
            "mlflow_service": {
                "tracking_uri": "${tmp_path:}/tracking/",
                "registry_uri": "${tmp_path:}/registry/",
            }
        }
    }
    """
    return config


# %% - Resolvers


@pytest.fixture(scope="session", autouse=True)
def tests_path_resolver(tests_path: str) -> str:
    """Register the tests path resolver with OmegaConf."""

    def resolver() -> str:
        """Get tests path."""
        return tests_path

    omegaconf.OmegaConf.register_new_resolver("tests_path", resolver, use_cache=True, replace=False)
    return tests_path


@pytest.fixture(scope="session", autouse=True)
def tmp_path_resolver(session_tmp_path: str) -> str:
    """Register the session-scoped tmp path resolver with OmegaConf."""

    def resolver() -> str:
        """Get session tmp data path."""
        return session_tmp_path

    omegaconf.OmegaConf.register_new_resolver("tmp_path", resolver, use_cache=False, replace=True)
    return session_tmp_path


# %% - Services


@pytest.fixture(scope="session", autouse=True)
def logger_service() -> T.Generator[services.LoggerService, None, None]:
    """Return and start the logger service."""
    service = services.LoggerService(colorize=False, diagnose=True)
    service.start()
    yield service
    service.stop()


@pytest.fixture
def logger_caplog(
    caplog: pl.LogCaptureFixture, logger_service: services.LoggerService
) -> T.Generator[pl.LogCaptureFixture, None, None]:
    """Extend pytest caplog fixture with the logger service (loguru)."""
    # https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
    logger = logger_service.logger()
    handler_id = logger.add(
        caplog.handler,
        level=0,
        format="{message}",
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


# @pytest.fixture(scope="session", autouse=True)
# def alerts_service() -> T.Generator[services.AlertsService, None, None]:
#     """Return and start the alerter service."""
#     service = services.AlertsService(enable=False)
#     service.start()
#     yield service
#     service.stop()


@pytest.fixture(scope="session", autouse=True)
def mlflow_service(session_tmp_path: str) -> T.Generator[services.MlflowService, None, None]:
    """Return and start the mlflow service."""
    service = services.MlflowService(
        tracking_uri=f"{session_tmp_path}/tracking/",
        registry_uri=f"{session_tmp_path}/registry/",
        experiment_name="Experiment-Testing",
        registry_name="Registry-Testing",
    )
    service.start()
    yield service
    service.stop()


# %% - Signatures
