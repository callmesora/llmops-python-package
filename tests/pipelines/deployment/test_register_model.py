import pytest

from llmops_project.pipelines.deployment.register_model import LogAndRegisterModelJob

# %% IMPORTS


# %% TESTS


@pytest.mark.parametrize(
    "registry_model_name, staging_alias, llm_model_code_path, llm_confs, vector_store_path",
    [
        (
            "test_model",
            "champion",
            "/src/llmops_project/models/chatbot_with_guardrails.py",
            "/confs/rag_chain_config.yaml",
            "http://localhost:6333",
        ),
        pytest.param(
            "invalid_model",
            "champion",
            "/invalid/path/to/model/code",
            "/invalid/path/to/config",
            "/invalid/path/to/vector/store",
            marks=pytest.mark.xfail(reason="Invalid paths", raises=Exception),
        ),
    ],
)
def test_log_and_register_model_job(
    mlflow_service,
    logger_service,
    registry_model_name: str,
    staging_alias: str,
    llm_model_code_path: str,
    llm_confs: str,
    vector_store_path: str,
):
    # Given: A LogAndRegisterModelJob instance with the provided parameters
    job = LogAndRegisterModelJob(
        registry_model_name=registry_model_name,
        staging_alias=staging_alias,
        llm_model_code_path=llm_model_code_path,
        llm_confs=llm_confs,
        vector_store_path=vector_store_path,
        mlflow_service=mlflow_service,
        logger_service=logger_service,
    )

    # When: The job is run
    with job as runner:
        out = runner.run()

    # Then: Verify the expected results
    assert set(out) == {
        "model_specs",
        "self",
        "llm_code_path",
        "logger",
        "client",
        "config_path",
        "run_id",
        "vector_store_path",
        "input_example",
        "script_dir",
    }

    # Verify  if model was registered by checking if the model version exists

    latest_version = out["client"].get_model_version_by_alias(registry_model_name, staging_alias)

    tags = out["client"].get_model_version(registry_model_name, latest_version.version).tags

    assert "passed_tests" in tags, "Tag 'passed_tests' does not exist."
