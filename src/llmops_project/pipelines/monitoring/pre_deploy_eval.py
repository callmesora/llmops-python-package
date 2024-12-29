import typing as T
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from llmops_project.pipelines import base


class EvaluateModelJob(base.Job):  # type: ignore[misc]
    """Job to Evaluate the challenger model based on a QA dataset.

    Attributes:
        KIND (Literal["EvaluateModelJob"]): The kind of job.
        qa_dataset_path (str): Path to the QA dataset.
        registry_model_name (str): Name of the model in the registry.
        alias (str): Alias of the model version.
        vector_store_path (str): Path to the vector store.
        metric_tresholds (dict[float]): Dictionary of metric thresholds.
    """

    KIND: T.Literal["EvaluateModelJob"] = "EvaluateModelJob"

    qa_dataset_path: str
    registry_model_name: str
    alias: str
    vector_store_path: str
    metric_tresholds: dict[str, float]

    def load_qa_dataset(self, data_path: str) -> pd.DataFrame:
        """Load the QA dataset from the specified path.

        Args:
            data_path (str): Path to the QA dataset.

        Returns:
            pd.DataFrame: The loaded QA dataset.
        """
        df = pd.read_csv(data_path)
        df = df.copy()
        df = df.rename(
            columns={
                "query": "inputs",
                "reference_answer": "ground_truth",
                "reference_contexts": "context",
            }
        )
        return df

    def generate_python_function_from_model(
        self, model_name: str, model_alias: str, vector_db_path: str
    ) -> T.Callable[[pd.DataFrame], pd.Series]:
        """Generate a Python function from the model.

        Args:
            model_name (str): Name of the model.
            model_alias (str): Alias of the model version.
            vector_db_path (str): Path to the vector store.

        Returns:
            Callable[[pd.DataFrame], pd.Series]: A function that takes a DataFrame of inputs and returns a Series of predictions.
        """
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.langchain.load_model(model_uri)

        def model_qa(inputs: pd.Series) -> pd.Series:
            answers = []
            for index, row in inputs.iterrows():
                question = {
                    "messages": [
                        {"role": "user", "content": f"{row['inputs']}"},
                    ],
                    "vector_store_path": vector_db_path,
                }
                answer = model.invoke(question)
                answers.append(answer["result"])
            return answers

        return model_qa

    def evaluate_model(self, eval_df: pd.DataFrame) -> mlflow.models.EvaluationResult:
        """Evaluate the model using the evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): DataFrame containing the evaluation data.

        Returns:
            mlflow.models.EvaluationResult: The evaluation results.
        """
        with mlflow.start_run():
            results = mlflow.evaluate(  # type: ignore
                data=eval_df[["inputs", "ground_truth", "predictions"]],
                targets="ground_truth",
                predictions="predictions",
                model_type="question-answering",
                evaluators=["default"],
            )
            return results

    def set_tag_for_model_evals(
        self, beats_baseline: bool, model_name: str, current_alias: str = "champion"
    ) -> None:
        """Set a tag for the model evaluations.

        Args:
            beats_baseline (bool): Whether the model meets the evaluation criteria.
            model_name (str): Name of the model.
            current_alias (str, optional): Alias of the current model version. Defaults to "champion".
        """
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(name=model_name, alias=current_alias)
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="meets_evaluation_criteria",
            value=beats_baseline,
        )

    @T.override
    def run(self) -> base.Locals:
        """Run the job to evaluate the model.

        Returns:
            base.Locals: The local variables after running the job.
        """
        # services
        logger = self.logger_service.logger()

        # Set up paths
        script_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)

        logger.info("Script Directory: {}", script_dir)
        data_path = str(script_dir + self.qa_dataset_path)

        logger.info("Loading QA dataset from {}", data_path)
        eval_df = self.load_qa_dataset(data_path)
        model = self.generate_python_function_from_model(
            self.registry_model_name, self.alias, self.vector_store_path
        )
        logger.info('Using Vector Store at "{}"', self.vector_store_path)

        logger.info("Running Predictions on the QA Dataset")
        eval_df["predictions"] = model(eval_df)

        logger.info("Evaluating the model")
        results = self.evaluate_model(eval_df)
        result_metrics = results.metrics

        metrics = [
            result_metrics["flesch_kincaid_grade_level/v1/mean"],
            result_metrics["ari_grade_level/v1/mean"],
        ]

        logger.info("Model Evaluation Metrics: {}", result_metrics)

        thresholds = [
            self.metric_tresholds["flesch_kincaid_grade_level_mean"],
            self.metric_tresholds["ari_grade_level_mean"],
        ]

        beats_baseline = True
        for metric, threshold in zip(metrics, thresholds):
            if metric < threshold:
                beats_baseline = False
                break

        logger.info(f"Model meets evaluation criteria: {beats_baseline}")

        self.set_tag_for_model_evals(
            beats_baseline, model_name=self.registry_model_name, current_alias=self.alias
        )
        logger.success("Model evaluation complete")

        return locals()


if __name__ == "__main__":
    from pathlib import Path

    from llmops_project import settings
    from llmops_project.io import configs

    script_dir = str(Path(__file__).parent.parent.parent.parent.parent)
    config_files = ["/monitoring.yaml"]

    file_paths = [script_dir + "/confs/" + file for file in config_files]

    files = [configs.parse_file(file) for file in file_paths]

    config = configs.merge_configs([*files])  # type: ignore
    config["job"]["KIND"] = "EvaluateModelJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
