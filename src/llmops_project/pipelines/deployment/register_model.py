# %% IMPORTS
import typing as T
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from pydantic import BaseModel, ValidationError

import llmops_project.io.services as services
from llmops_project.pipelines import base

logger = services.LoggerService().logger()


# %% Function to log the model to MLflow
def log_rag_model(
    model_path: str, config_path: str, input_example: T.Optional[Dict[str, Any]] = None
) -> str:
    # Load model configuration from the config file
    # Start an MLflow run and log the model
    logger.warning("Config_path")
    with mlflow.start_run(run_name="rag_with_guardrails") as run:
        mlflow.langchain.log_model(
            lc_model=model_path,  # Path to the chain code file
            model_config=config_path,  # Path to the chain configuration file
            artifact_path="chain",  # Required by MLflow
            code_paths=[
                config_path
            ],  # dependency definition included for the model to successfully import the implementation
            input_example=input_example,  # Input example for schema logging
            example_no_conversion=True,  # Use input_example directly as the chain's schema
        )
        return run.info.run_id  # Return the run ID for model registration


# %% Function to register the model in the MLflow model registry
def register_model(client: MlflowClient, run_id: str, model_name: str):
    model_uri = f"runs:/{run_id}/chain"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.success(
        f"Model registered successfully with name: {model_name}, version {result.version}"
    )

    client.set_registered_model_tag(name=model_name, key="model", value="claude3-haiku")


def load_model_by_alias(model_name, alias=None):
    # Construct the model URI using the alias if provided
    if alias:
        model_uri = f"models:/{model_name}@{alias}"
    else:
        model_uri = f"models:/{model_name}/latest"

    # Load the model
    model = mlflow.langchain.load_model(model_uri)

    return model


# Step 1: Define the expected output schema using Pydantic
class OutputSchema(BaseModel):
    result: str
    sources: List[Any]


# Step 2: Create a function to validate the output against the schema
def validate_output_schema(output: Dict[str, Any], schema: OutputSchema) -> bool:
    try:
        # Step 3: Validate the output against the schema model
        schema.model_validate(output)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False


def validate_model_signature(
    client: MlflowClient,
    model_name: str,
    vector_store_path: str,
    alias=None,
):
    """
    Validates the model signature by testing it against relevant and non-relevant dialogs.

    Args:
        model_name (str): The name of the model to validate.
        alias (str, optional): An alias for the model. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the model fails the schema or guardrail tests.

    This function performs the following steps:
        1. Loads the model using the provided name and alias.
        2. Invokes the model with relevant and non-relevant dialogs.
        3. Validates the model's output against a predefined schema.
        4. Checks if the model's output passes guardrail tests.
        5. Updates the model's tags in MLflow with the test results.
    """
    # script_dir = Path(__file__).resolve().parent.parent
    # project_root = script_dir.parent.parent.parent  # Adjusted to get to the project root as needed

    # vector_store_path = project_root / "faiss_db/"

    # Load Relevant Dialog and Non Relevant Dialog
    non_relevant_dialog = {  # This will test Guardrail
        "messages": [
            {"role": "user", "content": "What is the company's sick leave policy?"},
            {
                "role": "assistant",
                "content": "The company's sick leave policy allows employees to take a certain number of sick days per year. Please refer to the employee handbook for specific details and eligibility criteria.",
            },
            {"role": "user", "content": "What is the meaning of life?"},
        ],
        "vector_store_path": vector_store_path,
    }

    relevant_dialog = {  # This will test schema
        "messages": [
            {"role": "user", "content": "What is the company sick leave policy?"},
        ],
        "vector_store_path": vector_store_path,
    }

    model = load_model_by_alias(model_name, alias)

    non_relevant_result = model.invoke(non_relevant_dialog)
    relevant_result = model.invoke(relevant_dialog)

    logger.debug(f"Relevant Result: {relevant_result}")
    logger.debug(f"Non Relevant Result: {non_relevant_result}")
    # Validate the output against the schema
    is_schema_valid = validate_output_schema(relevant_result, OutputSchema)  # type: ignore
    if is_schema_valid:
        logger.success("Model Passsed Schema Tests")
    else:
        logger.error("Model Failed Schema Tests")

    # Validate Guardrail
    # Specific value to validate against
    guardrail_valid_output = {"result": "I cannot answer this question.", "sources": []}

    passed_guardrail_test = guardrail_valid_output == non_relevant_result
    if passed_guardrail_test:
        logger.success("Model Passsed Guadrail Tests")
    else:
        logger.error("Model Failed Guadrail Tests")

    if passed_guardrail_test and is_schema_valid:
        logger.success("Model Passed all tests")
        passed_tests = True
    else:
        logger.error("Model Failed tests")
        passed_tests = False

    # Update Model Tags

    filter_string = f"name = '{model_name}'"
    results = client.search_model_versions(filter_string=filter_string)
    latest_version = max(results, key=lambda mv: int(mv.version))

    client.set_model_version_tag(
        name=model_name, version=latest_version.version, key="passed_tests", value=str(passed_tests)
    )


def promote_model(client: MlflowClient, model_name: str, alias: str):
    # Get latest version
    filter_string = f"name = '{model_name}'"
    results = client.search_model_versions(filter_string=filter_string)
    latest_version = max(results, key=lambda mv: int(mv.version))
    tags = latest_version.tags

    if tags["passed_tests"].lower() == "true":
        client.set_registered_model_alias(
            name=model_name, alias=alias, version=latest_version.version
        )
    else:
        logger.error(
            "COULD NOT PROMOTE MODEL: MODEL FAILED TESTS OR IS NOTE BETTER THAN PREVIOUS MODEL"
        )


# %% Job class for logging and registering the RAG model
class LogAndRegisterModelJob(base.Job):  # type: ignore[misc]
    """Job to log and register the RAG model in MLflow.

    Parameters:
        run_config (services.MlflowService.RunConfig): mlflow run config.
    """

    KIND: T.Literal["LogAndRegisterModelJob"] = "LogAndRegisterModelJob"

    registry_model_name: str
    staging_alias: str = "champion"
    llm_model_code_path: str
    llm_confs: str
    vector_store_path: str

    @T.override
    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()

        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)

        logger.info(f"Logging Model Named {self.registry_model_name}")

        # Load Configuration
        script_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = str(str(script_dir) + self.llm_confs)
        llm_code_path = str(str(script_dir) + self.llm_model_code_path)
        vector_store_path = str(str(script_dir) + self.vector_store_path)

        logger.info(f"CONFIG PATH: {config_path}")

        model_specs = mlflow.models.ModelConfig(development_config=config_path)
        input_example = model_specs.get("input_example")

        run_id = log_rag_model(
            llm_code_path, config_path, input_example=input_example
        )  # Log the model and get the run ID

        logger.info(f"Registering Model Named {self.registry_model_name}")
        register_model(client, run_id, self.registry_model_name)  # Register the model

        logger.info(f"Validating Model Signature for {self.registry_model_name}")
        validate_model_signature(
            client,
            model_name=self.registry_model_name,
            vector_store_path=self.vector_store_path,
            alias=None,
        )

        promote_model(
            client=client, model_name=self.registry_model_name, alias="champion"
        )  # Promote model to champion if he passed the tests
        logger.info(f"Promoting Model Named {self.registry_model_name}  to {self.staging_alias}")
        logger.success("Model Registration complete")
        return locals()


if __name__ == "__main__":
    from pathlib import Path

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/deployment.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "LogAndRegisterModelJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
