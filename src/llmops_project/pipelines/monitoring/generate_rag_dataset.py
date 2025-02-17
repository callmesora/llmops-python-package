import typing as T
from pathlib import Path

import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.bedrock import Bedrock

from llmops_project.pipelines import base

nest_asyncio.apply()


# %% Job class for generating the RAG dataset
class GenerateRagDatasetJob(base.Job):  # type: ignore[misc]
    """Job to Generate RAG evaluation dataset from documents in the specified data path.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["GenerateRagDatasetJob"] = "GenerateRagDatasetJob"

    data_path: str
    qa_dataset_path_csv: str
    qa_dataset_path_json: str
    llm_model: str

    def generate_rag_dataset(
        self, data_path: str, final_dataset_csv_path: str, final_dataset_json_path: str, model: str
    ):
        """Generate a RAG dataset from documents in the specified data path.

        Args:
            data_path (str): Path to the directory containing the data.
            final_dataset_path (str): Path where the final dataset CSV will be saved.
            model (str): The model to be used for generating the dataset.
        """
        nest_asyncio.apply()
        logger = self.logger_service.logger()

        # Convert string paths to Path objects
        script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
        data_path = script_dir + self.data_path

        final_dataset_path = final_dataset_csv_path

        logger.info("Loading Data from ", data_path)
        # Load documents from the specified data path
        reader = SimpleDirectoryReader(data_path)
        documents = reader.load_data()

        logger.info("Loaded {} documents".format(len(documents)))

        # Initialize the LLM with the specified model
        llm = Bedrock(model=model, request_timeout=60.0)

        # Generate the dataset from the documents
        dataset_generator = RagDatasetGenerator.from_documents(
            documents,
            llm=llm,
            num_questions_per_chunk=2,
            show_progress=True,
        )

        # Generate the RAG dataset
        rag_dataset = dataset_generator.generate_dataset_from_nodes()

        # Convert the dataset to a pandas DataFrame and save it as a CSV
        df_dataset = rag_dataset.to_pandas()
        df_dataset.to_csv(final_dataset_path)

        # Save the dataset as a JSON file
        rag_dataset.save_json(final_dataset_json_path)

        logger.success("RAG dataset generated successfully and saved to {}", final_dataset_path)

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()

        # Set up paths
        # Ensure the paths are relative to this script's location
        script_dir = Path(__file__).resolve().parent.parent
        project_root = (
            script_dir.parent.parent.parent
        )  # Adjusted to get to the project root as needed

        data_path = str(project_root / self.data_path)
        final_dataset_path = str(project_root / self.qa_dataset_path_csv)
        final_dataset_json_path = str(project_root / self.qa_dataset_path_json)

        # Generate RAG Dataset
        logger.info("Generating RAG dataset from documents in {}", data_path)
        self.generate_rag_dataset(
            data_path, final_dataset_path, final_dataset_json_path, self.llm_model
        )

        logger.success("RAG dataset generated successfully")

        return locals()


if __name__ == "__main__":
    from pathlib import Path

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/generate_rag_dataset.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "GenerateRagDatasetJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
