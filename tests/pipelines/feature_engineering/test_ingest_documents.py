from unittest import mock

import pytest
from langchain_aws import BedrockEmbeddings

from llmops_project.io import services
from llmops_project.io.vector_db import QdrantVectorDB
from llmops_project.pipelines.feature_engineering.ingest_documents import IngestAndUpdateVectorDBJob

# %% IMPORTS


# %% TESTS


@pytest.fixture
def mock_bedrock_embeddings():
    return mock.Mock(spec=BedrockEmbeddings)


@pytest.fixture
def mock_qdrant_vector_db():
    return mock.Mock(spec=QdrantVectorDB)


@pytest.mark.parametrize(
    "embedding_model, collection_name, vector_store_path, document_path",
    [
        (
            "amazon.titan-embed-text-v1",
            "test_collection",
            "http://localhost:6333",
            "/tests/documents/",
        ),
        pytest.param(
            "invalid_model",
            "test_collection",
            "http://localhost:6333",
            "/tests/documents/",
            marks=pytest.mark.xfail(reason="Invalid embedding model", raises=Exception),
        ),
        pytest.param(
            "amazon.titan-embed-text-v1",
            "test_collection",
            "http://localhost:6333",
            "/invalid_path",
            marks=pytest.mark.xfail(reason=" Directory not found", raises=FileNotFoundError),
        ),
    ],
)
def test_ingest_and_update_vector_db_job(
    logger_service: services.LoggerService,
    embedding_model: str,
    collection_name: str,
    vector_store_path: str,
    document_path: str,
):
    job = IngestAndUpdateVectorDBJob(
        embedding_model=embedding_model,
        collection_name=collection_name,
        vector_store_path=vector_store_path,
        document_path=document_path,
        logger_service=logger_service,
    )

    with job as runner:
        result = runner.run()

    assert set(result.keys()) == {
        "self",
        "logger",
        "embeddings",
        "vector_db",
        "script_dir",
        "document_path",
    }

    # Try Querying the Qdrant Vector Store
    assert result["vector_db"].embeddings.model_id == embedding_model
    assert result["vector_db"].collection_name == collection_name

    query_results = result["vector_db"].query_database("What is the content of the documents?")
    for res in query_results:
        assert set(res.keys()) == {"score", "text", "source"}
        assert res["score"] is not None
        assert isinstance(res["text"], str)
        assert isinstance(res["source"], str)
