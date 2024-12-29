import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import litserve as ls
import mlflow
import mlflow.langchain
from pydantic import BaseModel

# ==== DEFINING INPUT / OUTPUT SCHEMAS WITH EXAMPLES ====


class MessageModel(BaseModel):
    role: str = "user"
    content: str = "What is the company's sick leave policy?"


# Input Format
class TextRequestModel(BaseModel):
    messages: List[MessageModel] = [
        MessageModel(role="user", content="What is the company's sick leave policy?"),
        MessageModel(
            role="assistant",
            content="The company's sick leave policy allows employees to take a certain number of sick days per year. Please refer to the employee handbook for specific details and eligibility criteria.",
        ),
        MessageModel(role="user", content="What is the meaning of life?"),
    ]
    vector_store_path: str = "http://host.docker.internal:6333"


# Output Format
class TextResponseModel(BaseModel):
    response: str = "The company's sick leave policy allows employees to take a certain number of sick days per year."
    source_documents: Any = None
    model_uri: str = "models:/rag-chatbot/latest"


class LangchainRAGAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and any required resources"""
        # Configuration - these could be moved to environment variables
        self.model_uri = "models:/rag-chatbot-with-guardrails/latest"
        self.cache_dir = "/tmp/mlflow_cache"

        # server uri
        tracking_uri: str = "http://127.0.0.1:5001"
        registry_uri: str = "http://127.0.0.1:5001"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        # Initialize cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load the chain
        self._initialize_chain()

    def _initialize_chain(self):
        """Load the chain with caching support"""
        try:
            cache_key = self.model_uri.replace("/", "_").replace(":", "_")
            cache_path = Path(self.cache_dir) / f"{cache_key}.json"

            if cache_path.exists():
                logging.info(f"Loading chain from cache: {cache_path}")
                self.chain = mlflow.langchain.load_model(self.model_uri)
            else:
                logging.info(f"Loading chain from MLflow: {self.model_uri}")
                self.chain = mlflow.langchain.load_model(self.model_uri)

        except Exception as e:
            logging.error(f"Error loading chain: {str(e)}")
            raise

    def decode_request(self, request: TextRequestModel) -> Dict[str, Any]:
        """Decode and validate the incoming request"""
        return {
            "messages": [message.dict() for message in request.messages],
            "vector_store_path": request.vector_store_path,
        }

    def predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using the loaded chain"""
        try:
            # Invoke the chain
            result = self.chain.invoke(request_data)

            # Handle different response formats
            if isinstance(result, dict):
                response = {
                    "response": result.get("result", result.get("response", str(result))),
                    "source_documents": result.get("sources"),
                }
            else:
                logging.warning(f"Mismatched response format: {type(result)} - {result}")
                response = {"response": str(result)}

            return response

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise

    def encode_response(self, prediction: Dict[str, Any]) -> TextResponseModel:
        """Encode the prediction result into the final response format"""
        return TextResponseModel(
            response=prediction.get("response"),
            source_documents=prediction.get("source_documents"),
            model_uri=self.model_uri,
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize and start the server
    api = LangchainRAGAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)

    ############################################# Run in terminal #######################################
    # litserve dockerize server.py --port 8000 --gpu
