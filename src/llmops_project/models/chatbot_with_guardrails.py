# mypy: ignore-errors
import json
import os
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlflow
from dotenv import load_dotenv
from guardrails import Guard
from guardrails.stores.context import get_call_kwarg
from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import pipeline

# Get the current working directory
script_dir = Path(os.getcwd())

# Navigate up to the parent folder (you can use .parent to go up one level)
parent_dir = script_dir.parent
grandparent_dir = parent_dir.parent  # Go up one more level

# Combine the path to reach the config directory
config_path = "rag_chain_config.yaml"

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config=config_path)

guardrail_config = model_config.get("guardrail_config")
llm_config = model_config.get("llm_config")
retriever_config = model_config.get("retriever_config")


# %% Setup Guardrails
# Import Guard and Validator


def get_topics_present(text: str, topics: List[str]) -> List[str]:
    """
    Determines which topics from a given list are present in the provided text using the ChatBedrock model.

    Args:
        text (str): The text in which to search for topics.
        topics (List[str]): A list of topics to check for presence in the text.

    Returns:
        List[str]: A list of topics that are present in the text. If no topics are found, returns an empty list.

    Example:
        >>> text = "Artificial intelligence and machine learning are transforming the tech industry."
        >>> topics = ["artificial intelligence", "machine learning", "blockchain"]
        >>> get_topics_present(text, topics)
        ['artificial intelligence', 'machine learning']
    """
    # Initialize the ChatBedrock model

    model = ChatBedrock(
        model_id=guardrail_config["model"],
        model_kwargs=dict(temperature=0.01),
    )

    # Prepare the messages for the model
    messages = [
        {
            "role": "system",
            "content": "Do not Output python functions. Given a conversation and a list of topics, return a valid JSON list of which topics are present in the final user question. If none, just return an empty list.",
        },
        {
            "role": "user",
            "content": f"""
                Given a text containing a user question, a chat_history and a list of topics, return a valid JSON list of which topics are present in the user question.
                Consider the entire history of the chat to determine the topics present in the user question. If the user question is unrelated with the chat history, consider only the user question.
                If none, just return an empty list.

                Output Format:
                -------------
                {{
                    "topics_present": []
                }}

                Text:
                ----
                "{text}"

                Topics:
                ------
                {topics}

                Result:
                ------
            """,
        },
    ]

    # Invoke the model with the prepared messages
    response = json.loads(model.invoke(messages).content)
    return response["topics_present"]


def format_chat_history(messages):
    chat_history = ""
    for message in messages[:-1]:
        role = message["role"]
        content = message["content"]
        if role == "user":
            chat_history += f"User: {content}\n"
        elif role == "assistant":
            chat_history += f"Assistant: {content}\n"

    last_question = messages[-1]["content"]

    prompt = f"""You have been assigned with a task, based on the chat history below and the last user query.

    Chat history: {chat_history.strip()}

    Question: {last_question}"""

    return prompt


# Set Validation Function
def guardrail_the_input(messages):
    """
    Validates the input messages using guardrails.

    Args:
        messages (list): A list of message dictionaries representing the chat history.

    Returns:
        bool: True if the validation passes, False otherwise.
    """
    prompt = format_chat_history(messages)
    return guard.validate(prompt).validation_passed


# %% Setup Topic Guardrails


@register_validator(
    name="tryolabs/restricttotopic", data_type="string", has_guardrails_endpoint=True
)
class RestrictToTopic(Validator):
    """Checks if text's main topic is specified within a list of valid topics
    and ensures that the text is not about any of the invalid topics.

    This validator accepts at least one valid topic and an optional list of
    invalid topics.

    Default behavior first runs a Zero-Shot model, and then falls back to
    ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident
    in the topic classification (score < 0.5).

    In our experiments this LLM fallback increases accuracy by 15% but also
    increases latency (more than doubles the latency in the worst case).

    Both the Zero-Shot classification and the GPT classification may be toggled.

    **Key Properties**

    | Property                      | Description                              |
    | ----------------------------- | ---------------------------------------- |
    | Name for `format` attribute   | `tryolabs/restricttotopic`               |
    | Supported data types          | `string`                                 |
    | Programmatic fix              | Removes lines with off-topic information |

    Args:
        valid_topics (List[str]): topics that the text should be about
            (one or many).
        invalid_topics (List[str], Optional, defaults to []): topics that the
            text cannot be about.
        device (Optional[Union[str, int]], Optional, defaults to -1): Device ordinal for
            CPU/GPU supports for Zero-Shot classifier. Setting this to -1 will leverage
            CPU, a positive will run the Zero-Shot model on the associated CUDA
            device id.
        model (str, Optional, defaults to 'facebook/bart-large-mnli'): The
            Zero-Shot model that will be used to classify the topic. See a
            list of all models here:
            https://huggingface.co/models?pipeline_tag=zero-shot-classification
        llm_callable (Union[str, Callable, None], Optional, defaults to
            'gpt-4o'): Either the name of the OpenAI model, or a callable
            that takes a prompt and returns a response.
        disable_classifier (bool, Optional, defaults to False): controls whether
            to use the Zero-Shot model. At least one of disable_classifier and
            disable_llm must be False.
        classifier_api_endpoint (str, Optional, defaults to None): An API endpoint
            to recieve post requests that will be used when provided. If not provided, a
            local model will be initialized.
        disable_llm (bool, Optional, defaults to False): controls whether to use
            the LLM fallback. At least one of disable_classifier and
            disable_llm must be False.
        zero_shot_threshold (float, Optional, defaults to 0.5): The threshold used to
            determine whether to accept a topic from the Zero-Shot model. Must be
            a number between 0 and 1.
        llm_threshold (int, Optional, defaults to 3): The threshold used to determine
        if a topic exists based on the provided llm api. Must be between 0 and 5.
    """

    def __init__(
        self,
        valid_topics: List[str],
        invalid_topics: Optional[List[str]] = [],
        device: Optional[Union[str, int]] = -1,
        model: Optional[str] = "facebook/bart-large-mnli",
        llm_callable: Union[str, Callable, None] = None,
        disable_classifier: Optional[bool] = False,
        classifier_api_endpoint: Optional[str] = None,
        disable_llm: Optional[bool] = False,
        on_fail: Optional[Callable[..., Any]] = None,
        zero_shot_threshold: Optional[float] = 0.5,
        llm_threshold: Optional[int] = 3,
        **kwargs,
    ):
        super().__init__(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            device=device,
            model=model,
            disable_classifier=disable_classifier,
            classifier_api_endpoint=classifier_api_endpoint,
            disable_llm=disable_llm,
            llm_callable=llm_callable,
            on_fail=on_fail,
            zero_shot_threshold=zero_shot_threshold,
            llm_threshold=llm_threshold,
            **kwargs,
        )
        self._valid_topics = valid_topics
        if invalid_topics is None:
            self._invalid_topics = []
        else:
            self._invalid_topics = invalid_topics

        self._device = str(device).lower() if str(device).lower() in ["cpu", "mps"] else int(device)
        self._model = model
        self._disable_classifier = disable_classifier
        self._disable_llm = disable_llm
        self._classifier_api_endpoint = classifier_api_endpoint

        self._zero_shot_threshold = zero_shot_threshold
        if self._zero_shot_threshold < 0 or self._zero_shot_threshold > 1:
            raise ValueError("zero_shot_threshold must be a number between 0 and 1")

        self._llm_threshold = llm_threshold
        if self._llm_threshold < 0 or self._llm_threshold > 5:
            raise ValueError("llm_threshold must be a number between 0 and 5")
        self.set_callable(llm_callable)

        if self._classifier_api_endpoint is None and self.use_local:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._model,
                device=self._device,
                hypothesis_template="This example has to do with topic {}.",
                multi_label=True,
            )
        else:
            # TODO api endpoint
            ...

    def get_topics_ensemble(self, text: str, candidate_topics: List[str]) -> List[str]:
        """Finds the topics in the input text based on if it is determined by the zero
        shot model or the llm.

        Args:
            text (str): The input text to find categories from
            candidate_topics (List[str]): The topics to search for in the input text

        Returns:
            List[str]: The found topics
        """
        # Find topics based on zero shot model
        zero_shot_topics = self._inference(
            {"text": text, "valid_topics": candidate_topics, "invalid_topics": []}
        )

        # Find topics based on llm
        llm_topics = self.get_topics_llm(text, candidate_topics)

        return list(set(zero_shot_topics + llm_topics))

    def get_topics_llm(self, text: str, candidate_topics: List[str]) -> List[str]:
        """Returns a list of the topics identified in the given text using an LLM
        callable

        Args:
            text (str): The input text to classify topics.
            candidate_topics (List[str]): The topics to identify if present in the text.

        Returns:
            List[str]: The topics found in the input text.
        """
        llm_topics = self.call_llm(text, candidate_topics)
        found_topics = []
        for llm_topic in llm_topics:
            if llm_topic in candidate_topics:
                found_topics.append(llm_topic)
        return found_topics

    def get_client_args(self) -> Tuple[Optional[str], Optional[str]]:
        """Returns neccessary data for api calls.

        Returns:
            str: api key
        """

        load_dotenv()

        api_key = get_call_kwarg("api_key") or os.environ.get("OPENAI_API_KEY")
        api_base = get_call_kwarg("api_base") or os.environ.get("OPENAI_API_BASE")

        return (api_key, api_base)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call_llm(self, text: str, topics: List[str]) -> str:
        """Call the LLM with the given prompt.

        Expects a function that takes a string and returns a string.
        Args:
            text (str): The input text to classify using the LLM.
            topics (List[str]): The list of candidate topics.
        Returns:
            response (str): String representing the LLM response.
        """
        return self._llm_callable(text, topics)

    def set_callable(self, llm_callable: Union[str, Callable, None]) -> None:
        """Set the LLM callable.

        Args:
            llm_callable: Either the name of the OpenAI model, or a callable that takes
                a prompt and returns a response.
        """

        if llm_callable is None:
            llm_callable = "gpt-4o"

        if isinstance(llm_callable, str):
            if llm_callable not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
                raise ValueError(
                    "llm_callable must be one of 'gpt-3.5-turbo', 'gpt-4', or 'gpt-4o'"
                    "If you want to use a custom LLM, please provide a callable."
                    "Check out ProvenanceV1 documentation for an example."
                )

            def openai_callable(text: str, topics: List[str]) -> str:
                api_key, api_base = self.get_client_args()
                client = OpenAI(api_key=api_key, base_url=api_base)
                response = client.chat.completions.create(
                    model=llm_callable,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                                Given a text and a list of topics, return a valid json list of which topics are present in the text. If none, just return an empty list.

                                Output Format:
                                -------------
                                "topics_present": []

                                Text:
                                ----
                                "{text}"

                                Topics:
                                ------
                                {topics}

                                Result:
                                ------ """,
                        },
                    ],
                )
                return json.loads(response.choices[0].message.content)["topics_present"]

            self._llm_callable = openai_callable
        elif isinstance(llm_callable, Callable):
            self._llm_callable = llm_callable
        else:
            raise ValueError("llm_callable must be a string or a Callable")

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = {}) -> ValidationResult:
        """Validates that a string contains at least one valid topic and no invalid topics.

        Args:
            value (str): The given string to classify
            metadata (Optional[Dict[str, Any]], optional): Dictionary containing valid and invalid topics. Defaults to {}.


        Raises:
            ValueError: If a topic is invalid and valid
            ValueError: If no valid topics are set
            ValueError: If there is no llm or zero shot classifier set

        Returns:
            ValidationResult: PassResult if a topic is restricted and valid,
            FailResult otherwise
        """
        valid_topics = set(metadata.get("valid_topics", self._valid_topics))
        invalid_topics = set(metadata.get("invalid_topics", self._invalid_topics))
        all_topics = list(valid_topics | invalid_topics)

        # throw if valid and invalid topics are empty
        if not valid_topics:
            raise ValueError("`valid_topics` must be set and contain at least one topic.")

        # throw if valid and invalid topics are not disjoint
        if bool(valid_topics.intersection(invalid_topics)):
            raise ValueError("A topic cannot be valid and invalid at the same time.")

        model_input = {
            "text": value,
            "valid_topics": valid_topics,
            "invalid_topics": invalid_topics,
        }

        # Ensemble method
        if not self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_ensemble(value, all_topics)
        # LLM Classifier Only
        elif self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_llm(value, all_topics)
        # Zero Shot Classifier Only
        elif not self._disable_classifier and self._disable_llm:
            found_topics = self._inference(model_input)
        else:
            raise ValueError("Either classifier or llm must be enabled.")

        # Determine if valid or invalid topics were found
        invalid_topics_found = []
        valid_topics_found = []
        for topic in found_topics:
            if topic in valid_topics:
                valid_topics_found.append(topic)
            elif topic in invalid_topics:
                invalid_topics_found.append(topic)

        error_spans = []

        # Require at least one valid topic and no invalid topics
        if invalid_topics_found:
            for topic in invalid_topics_found:
                error_spans.append(
                    ErrorSpan(
                        start=value.find(topic),
                        end=value.find(topic) + len(topic),
                        reason=f"Text contains invalid topic: {topic}",
                    )
                )
            return FailResult(
                error_message=f"Invalid topics found: {invalid_topics_found}",
                error_spans=error_spans,
            )
        if not valid_topics_found:
            return FailResult(
                error_message="No valid topic was found.",
                error_spans=[
                    ErrorSpan(start=0, end=len(value), reason="No valid topic was found.")
                ],
            )
        return PassResult()

    def _inference_local(self, model_input: Any) -> Any:
        """Local inference method for the restrict-to-topic validator."""
        text = model_input["text"]
        candidate_topics = model_input["valid_topics"] + model_input["invalid_topics"]

        result = self._classifier(text, candidate_topics)
        topics = result["labels"]
        scores = result["scores"]
        found_topics = []
        for topic, score in zip(topics, scores):
            if score > self._zero_shot_threshold:
                found_topics.append(topic)
        return found_topics

    def _inference_remote(self, model_input: Any) -> Any:
        """Remote inference method for the restrict-to-topic validator."""
        request_body = {
            "inputs": [
                {"name": "text", "shape": [1], "data": [model_input["text"]], "datatype": "BYTES"},
                {
                    "name": "candidate_topics",
                    "shape": [
                        len(model_input["valid_topics"]) + len(model_input["invalid_topics"])
                    ],
                    "data": list(model_input["valid_topics"]) + list(model_input["invalid_topics"]),
                    "datatype": "BYTES",
                },
                {
                    "name": "zero_shot_threshold",
                    "shape": [1],
                    "data": [self._zero_shot_threshold],
                    "datatype": "FP32",
                },
            ]
        }

        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)

        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        return response["outputs"][0]["data"]


# Setup Guard
guard = Guard().use(
    RestrictToTopic(
        valid_topics=guardrail_config["topics"]["valid"],
        invalid_topics=guardrail_config["topics"]["invalid"],
        llm_callable=get_topics_present,
        disable_classifier=True,  # Choose to use classifier or LLM
        disable_llm=False,
        on_fail="noop",
    )
)

# %% Setup Chatbot Chainß


# The question is the last entry of the history
def extract_question(input: List[Dict[str, Any]]) -> str:
    """
    Extract the question from the input.

    Args:
        input (list[dict]): The input containing chat messages.

    Returns:
        str: The extracted question.
    """
    return input[-1]["content"]


# The history is everything before the last question
def extract_history(input: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract the chat history from the input.

    Args:
        input (list[dict]): The input containing chat messages.

    Returns:
        list[dict]: The extracted chat history.
    """
    return input[:-1]


chat_model = ChatBedrock(
    model_id=llm_config["llm_model"],
    model_kwargs=dict(temperature=0.01),
)


def get_retriever(
    path: str = "http://localhost:6333",
):
    """
    Get the retriever.

    Args:
        path (str, optional): The path to the vector store. Defaults to None.

    Returns:
        Any: The retriever.
    """
    embeddings = BedrockEmbeddings()
    api_key = os.getenv("QDRANT_API_KEY")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=retriever_config.get("collection_name", "default"),
        url=path,
        api_key=api_key if api_key else None,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": retriever_config.get("parameters")["k"]}
    )

    return retriever


# Setup Prompt to re-write query from chat history context
generate_query_to_retrieve_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=llm_config["query_rewriter_prompt_template"],
)


# Setup query rewriter chain
generate_query_to_retrieve_context_chain = {
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
} | RunnableBranch(  # Augment query only when there is a chat history
    (
        lambda x: x["chat_history"],
        generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser(),
    ),
    (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
    RunnableLambda(lambda x: x["question"]),
)  # type: ignore


question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=llm_config.get("llm_prompt_template"),  # Add Question with History and Context Prompt
)


def format_context(docs: List[Document]) -> str:
    """
    Format the context from a list of documents.

    Args:
        docs (list[Document]): A list of documents.

    Returns:
        str: A formatted string containing the content of the documents.
    """
    return "\n\n".join([d.page_content for d in docs])


def extract_source_urls(docs: List[Document]) -> List[str]:
    """
    Extract source URLs from a list of documents.

    Args:
        docs (list[Document]): A list of documents.

    Returns:
        list[str]: A list of source URLs extracted from the documents' metadata.
    """
    return [d.metadata[retriever_config.get("schema")["document_uri"]] for d in docs]


relevant_question_chain = (
    RunnablePassthrough()  # type: ignore
    | {
        "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser(),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
        "vector_store_path": itemgetter("vector_store_path"),
    }
    | {
        "relevant_docs": itemgetter("relevant_docs"),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
        "vector_store_path": itemgetter("vector_store_path"),
    }
    | RunnableLambda(
        lambda x: {
            "relevant_docs": get_retriever(x["vector_store_path"]).get_relevant_documents(
                x["relevant_docs"]
            ),
            "chat_history": x["chat_history"],
            "question": x["question"],
            "vector_store_path": x["vector_store_path"],
        }
    )
    | {
        "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
        "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
        "chat_history": itemgetter("chat_history"),
        "question": itemgetter("question"),
    }
    | {"prompt": question_with_history_and_context_prompt, "sources": itemgetter("sources")}
    | {
        "result": itemgetter("prompt") | chat_model | StrOutputParser(),
        "sources": itemgetter("sources"),
    }
)


irrelevant_question_chain = RunnableLambda(
    lambda x: {"result": llm_config.get("llm_refusal_fallback_answer"), "sources": []}
)
branch_node = RunnableBranch(
    (lambda x: x["guardrails_validated"] == True, relevant_question_chain),  # noqa: E712
    (lambda x: x["guardrails_validated"] == False, irrelevant_question_chain),  # noqa: E712
    irrelevant_question_chain,  # Default
)  # type: ignore


full_chain = {
    "guardrails_validated": itemgetter("messages") | RunnableLambda(guardrail_the_input),
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    "vector_store_path": itemgetter("vector_store_path"),
} | branch_node  # type: ignore


## Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=full_chain)  # type: ignore
