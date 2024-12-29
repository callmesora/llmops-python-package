# %% IMPORTS
import os
import typing as T
from pathlib import Path

import dotenv
from langchain_aws import BedrockEmbeddings

from llmops_project.io.vector_db import QdrantVectorDB

# import faiss
from llmops_project.pipelines import base


# %% Job class for creating the vector database
class CreateVectorDBJob(base.Job):  # type: ignore[misc]
    """Job to create an empty FAISS vector store.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["CreateVectorDBJob"] = "CreateVectorDBJob"

    embedding_model: str
    collection_name: str
    vector_store_path: str

    @T.override
    def run(self) -> base.Locals:
        # Setup services
        # services
        # - logger
        logger = self.logger_service.logger()

        # Run the main pipeline function to create the empty vector database
        # Load .env file on the grandparent folder
        script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
        dotenv.load_dotenv(script_dir + "/.env")

        embeddings = BedrockEmbeddings(model_id=self.embedding_model)

        vector_db = QdrantVectorDB(
            embeddings_model=embeddings,
            collection_name=self.collection_name,
            url=self.vector_store_path,
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=1536,
        )

        logger.info("Initializing empty Qdrant Collection vector store...")

        try:
            vector_db.create_vector_db()
        except Exception as e:
            if "409" in str(e):
                logger.warning(f"Collection {self.collection_name} already exists")
            else:
                raise e

        logger.success(
            f"{vector_db.__class__.__name__} vector store created successfully on path {self.vector_store_path}"
        )
        return locals()


if __name__ == "__main__":
    # Test the pipeline

    from pathlib import Path

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/rag_chain_config.yaml", "/feature_eng.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "CreateVectorDBJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
