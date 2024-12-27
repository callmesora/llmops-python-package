from unittest import mock

import pytest
from langchain_aws import BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore

from llmops_project.io import services
from llmops_project.io.vector_db import QdrantVectorDB
from llmops_project.pipelines.feature_engineering.create_vector_db import CreateVectorDBJob

# %% IMPORTS


# %% TESTS


@pytest.fixture
def mock_bedrock_embeddings():
    return mock.Mock(spec=BedrockEmbeddings)


@pytest.fixture
def mock_qdrant_vector_db():
    return mock.Mock(spec=QdrantVectorDB)


@pytest.mark.parametrize(
    "embedding_model, collection_name, vector_store_path",
    [
        ("amazon.titan-embed-text-v1", "test_collection", "http://localhost:6333"),
        pytest.param(
            "amazon.titan-embed-text-v1",
            "test_collection",
            "http://localhost:6334",
            marks=pytest.mark.xfail(reason="Invalid localhost port", raises=Exception),
        ),
    ],
)
def test_create_vector_db_job(
    logger_service: services.LoggerService,
    mlflow_service: services.MlflowService,
    embedding_model: str,
    collection_name: str,
    vector_store_path: str,
):
    job = CreateVectorDBJob(
        embedding_model=embedding_model,
        collection_name=collection_name,
        vector_store_path=vector_store_path,
        logger_service=logger_service,
    )

    with job as runner:
        out = runner.run()

    assert set(out) == {"self", "logger", "embeddings", "vector_db", "script_dir"}

    # Vector Db
    assert out["vector_db"].embeddings.model_id == embedding_model
    assert out["vector_db"].collection_name == collection_name

    assert out["embeddings"].model_id == embedding_model

    try:
        QdrantVectorStore.from_existing_collection(
            embedding=out["embeddings"],
            collection_name=collection_name,
            url=vector_store_path,
        )
    except Exception as e:
        pytest.fail(f"Failed to create QdrantVectorStore: {e}")
