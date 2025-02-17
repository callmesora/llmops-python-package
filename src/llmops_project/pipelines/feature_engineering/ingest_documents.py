# %% IMPORTS
import os
import typing as T
from pathlib import Path

import dotenv
from langchain_aws import BedrockEmbeddings

from llmops_project.io import services
from llmops_project.io.vector_db import QdrantVectorDB
from llmops_project.pipelines import base

logger = services.LoggerService().logger()


# %% Job class for ingesting documents and updating the vector database
class IngestAndUpdateVectorDBJob(base.Job):  # type: ignore[misc]
    """Job to ingest documents and update the FAISS vector store.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["IngestAndUpdateVectorDBJob"] = "IngestAndUpdateVectorDBJob"

    embedding_model: str
    vector_store_path: str
    collection_name: str
    document_path: str

    @T.override
    def run(self) -> base.Locals:
        # Setup services
        # services
        # - logger
        logger = self.logger_service.logger()

        script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
        document_path = script_dir + self.document_path
        dotenv.load_dotenv(script_dir + "/.env")

        logger.info(f"Loading Documents from {document_path}...")

        embeddings = BedrockEmbeddings(model_id=self.embedding_model)

        vector_db = QdrantVectorDB(
            embeddings_model=embeddings,
            collection_name=self.collection_name,
            url=self.vector_store_path,
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=1536,
        )

        logger.info(
            f"Ingesting documents and updating the {vector_db.__class__.__name__} vector store..."
        )

        vector_db.ingest_documents(document_path)

        logger.success("Documents ingested and vector store updated successfully")
        # test_vectordb()
        return locals()


if __name__ == "__main__":
    # Test the pipeline

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/rag_chain_config.yaml", "/feature_eng.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "IngestAndUpdateVectorDBJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
