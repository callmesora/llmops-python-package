from pathlib import Path

import pytest

from llmops_project.pipelines.monitoring.generate_rag_dataset import GenerateRagDatasetJob


@pytest.mark.parametrize(
    "data_path, qa_dataset_path_csv, qa_dataset_path_json, llm_model",
    [
        (
            "/tests/documents/",
            "data/qa_dataset.csv",
            "data/qa_dataset.json",
            "anthropic.claude-3-haiku-20240307-v1:0",
        ),
        pytest.param(
            "/invalid_path",
            "data/qa_dataset.csv",
            "data/qa_dataset.json",
            "anthropic.claude-3-haiku-20240307-v1:0",
            marks=pytest.mark.xfail(reason="Invalid data path", raises=Exception),
        ),
    ],
)
def test_generate_rag_dataset_job(
    logger_service,
    data_path: str,
    qa_dataset_path_csv: str,
    qa_dataset_path_json: str,
    llm_model: str,
):
    # Given: A GenerateRagDatasetJob instance with the provided parameters
    job = GenerateRagDatasetJob(
        data_path=data_path,
        qa_dataset_path_csv=qa_dataset_path_csv,
        qa_dataset_path_json=qa_dataset_path_json,
        llm_model=llm_model,
        logger_service=logger_service,
    )

    # When: The job is run
    with job as runner:
        out = runner.run()

    # Then: Verify the expected results
    assert set(out) == {
        "self",
        "data_path",
        "final_dataset_path",
        "final_dataset_json_path",
        "logger",
        "script_dir",
        "project_root",
    }

    # Verify if the CSV and JSON files are created
    assert Path(out["final_dataset_path"]).exists(), "CSV file was not created."
    assert Path(out["final_dataset_json_path"]).exists(), "JSON file was not created."
