import pytest

from llmops_project.pipelines.deployment.deploy_model import DeployModelJob

# %% IMPORTS


# %% TESTS


@pytest.mark.parametrize(
    "registry_model_name, staging_alias, production_alias",
    [
        ("test_model", "champion", "production"),
        pytest.param(
            "invalid_model",
            "champion",
            "production",
            marks=pytest.mark.xfail(reason="Invalid model name", raises=Exception),
        ),
    ],
)
def test_deploy_model_job(
    mlflow_service,
    logger_service,
    registry_model_name: str,
    staging_alias: str,
    production_alias: str,
):
    job = DeployModelJob(
        registry_model_name=registry_model_name,
        staging_alias=staging_alias,
        production_alias=production_alias,
        mlflow_service=mlflow_service,
        logger_service=logger_service,
    )

    with job as runner:
        result = runner.run()

    assert set(result.keys()) == {
        "self",
        "logger",
        "client",
    }

    model_version = result["client"].get_model_version_by_alias(
        name=registry_model_name, alias=production_alias
    )
    tags = model_version.tags
    assert tags["passed_tests"] == "True"
