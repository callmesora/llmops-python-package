import pytest

from llmops_project.pipelines.monitoring.pre_deploy_eval import EvaluateModelJob

# %% TESTS


@pytest.mark.parametrize(
    "qa_dataset_path, registry_model_name, alias, vector_store_path, metric_tresholds, expect_failure",
    [
        (
            "/data/qa_dataset.csv",
            "test_model",
            "champion",
            "http://localhost:6333",
            {"flesch_kincaid_grade_level_mean": 5.0, "ari_grade_level_mean": 5.0},
            False,
        ),
        pytest.param(
            "/invalid/path/to/qa_dataset.csv",
            "test_model",
            "champion",
            "/invalid/path/to/vector_store",
            {"flesch_kincaid_grade_level_mean": 5.0, "ari_grade_level_mean": 5.0},
            True,
            marks=pytest.mark.xfail(reason="Invalid paths", raises=Exception),
        ),
    ],
)
def test_evaluate_model_job(
    mlflow_service,
    logger_service,
    qa_dataset_path: str,
    registry_model_name: str,
    alias: str,
    vector_store_path: str,
    metric_tresholds: dict,
    expect_failure: bool,
):
    # Given: An EvaluateModelJob instance with the provided parameters
    job = EvaluateModelJob(
        qa_dataset_path=qa_dataset_path,
        registry_model_name=registry_model_name,
        alias=alias,
        vector_store_path=vector_store_path,
        metric_tresholds=metric_tresholds,
        mlflow_service=mlflow_service,
        logger_service=logger_service,
    )

    # When: The job is run
    with job as runner:
        out = runner.run()

    # Then: Verify the expected results
    if expect_failure:
        assert "eval_df" not in out, "Evaluation DataFrame should not be present."
    else:
        assert "eval_df" in out, "Evaluation DataFrame not found in output."
        assert "results" in out, "Results not found in output."

        # Verify the output variables
        assert set(out) == {
            "logger",
            "script_dir",
            "data_path",
            "eval_df",
            "model",
            "results",
            "result_metrics",
            "metrics",
            "thresholds",
            "beats_baseline",
            "self",
            "threshold",
            "metric",
        }
