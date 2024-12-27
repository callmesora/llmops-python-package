import json
import time
import typing as T
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from llmops_project.pipelines import base


def filter_generations(df):
    return df[
        df["response"].apply(
            lambda x: "generations" not in json.loads(x) if pd.notnull(x) else True
        )
    ]


def extract_answer(data):
    if data:
        data_dict = json.loads(data)
        if "result" in data_dict:
            return data_dict["result"]
    return None


def extract_last_message_content(request):
    return json.loads(request)["messages"][-1]["content"]


def create_gauge_chart(value1, title1, value2, title2):
    # Create a subplot figure with two columns
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])

    # Add the first gauge chart
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=value1,
            title={"text": title1},
            gauge={"axis": {"range": [None, 18]}},
        ),
        row=1,
        col=1,
    )

    # Add the second gauge chart
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=value2,
            title={"text": title2},
            gauge={"axis": {"range": [None, 100]}},
        ),
        row=1,
        col=2,
    )

    # Update layout
    fig.update_layout(height=400, width=800)

    # Show figure
    # fig.show()

    return fig


class MonitoringEvalJob(base.Job):  # type: ignore[misc]
    """Job to Evaluate the challenger model based on a QA dataset."""

    KIND: T.Literal["MonitoringEvalJob"] = "MonitoringEvalJob"

    trace_experiment_name: str
    monitoring_experiment_name: str
    filter_string: Optional[str] = None

    @T.override
    def run(self) -> base.Locals:
        """Run the job to evaluate the model.

        Returns:
            base.Locals: The local variables after running the job.
        """

        # services
        # - logger
        logger = self.logger_service.logger()

        # - mlflow
        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.trace_experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            logger.info(f"Experiment ID: {experiment_id}")
        else:
            logger.error("Experiment with the traces not found.")
            return locals()  # Add return statement here

        # Set the filter string to only include runs from the last week
        if self.filter_string is None:
            one_week_ago = int((time.time() - 7 * 24 * 60 * 60) * 1000)  # Convert to milliseconds
            filter_string = f"trace.timestamp_ms > {one_week_ago}"
            logger.success("Monitoring traces from the last week")

        else:
            filter_string = self.filter_string

        # Search all the traces in the experiment that match the filter string
        traces_df = mlflow.search_traces(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=2000,
        )

        # Filter error traces
        traces_df = traces_df[traces_df["status"] != "TraceStatus.ERROR"]
        traces_df = filter_generations(traces_df)

        # Extract the answer and question from the request and response
        traces_df["answer"] = traces_df["response"].apply(extract_answer)
        traces_df["question"] = traces_df["request"].apply(extract_last_message_content)

        #  Create a DataFrame with the inputs and predictions
        eval_df = traces_df[["question", "answer"]]
        eval_df = eval_df.rename(columns={"question": "inputs", "answer": "predictions"})

        # remove predictions with None values
        eval_df = eval_df.dropna()

        # Get the current week number
        current_week = time.strftime("CW%U")

        mlflow.set_experiment(self.monitoring_experiment_name)

        logger.info(
            "Monitoring results to be logged in experiment: {}", self.monitoring_experiment_name
        )

        answer_relevance = mlflow.metrics.genai.answer_relevance(  # Compares input with predictions to check if its relevant (good for monitoring)
            model="bedrock:/anthropic.claude-3-haiku-20240307-v1:0",
            parameters={
                "temperature": 0,
                "anthropic_version": "bedrock-2023-05-31",
            },
        )

        with mlflow.start_run(run_name=current_week):
            results = mlflow.evaluate(  # type: ignore
                data=eval_df[["inputs", "predictions"]],
                predictions="predictions",
                model_type="text",
                evaluators=["default"],
                extra_metrics=[answer_relevance],
            )

            toxicity_score = results.metrics["toxicity/v1/mean"]
            # Calculate non-toxicity score
            non_toxicity_score = "{:.2f}".format((1 - toxicity_score) * 100)
            readability_score = "{:.2f}".format(
                results.metrics["flesch_kincaid_grade_level/v1/mean"]
            )
            logger.info("Non Toxicity Score: {}", non_toxicity_score)
            logger.info("Readability Score: {}", readability_score)

            guage = create_gauge_chart(
                float(readability_score),
                "English Readability score",
                float(non_toxicity_score),
                "Non Toxicity Score",
            )
            mlflow.log_figure(guage, "gauge_chart.png")

        logger.success("Model Monitoring completed successfully.")

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
    config["job"]["KIND"] = "MonitoringEvalJob"  # type: ignore

    object_ = configs.to_object(config)  # python object

    setting = settings.MainSettings.model_validate(object_)

    with setting.job as runner:
        runner.run()
