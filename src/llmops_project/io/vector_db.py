# Create Abstract class to Create Delete and Ingest documents to Vector DB
import os
from abc import ABC, abstractmethod
from typing import List, Optional

# Import utility functions
from langchain_aws import BedrockEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class VectorDB(ABC):
    """Vector Database Abstract Class"""

    def __init__(self, embedding_model: str, embedding_model_size: int, vector_store_path: str):
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.embedding_model_size = embedding_model_size

    @abstractmethod
    def create_vector_db(self):
        pass

    @abstractmethod
    def delete_vector_db(self):
        pass

    @abstractmethod
    def ingest_documents(self, document_path: str):
        pass


# %% FAISS Vector Database Class


# Implement Faiss Vector Database Class
class FAISSVectorDB(VectorDB):
    """FAISS Vector Database Class"""

    def __init__(
        self,
        embedding_model: str = "amazon.titan-embed-text-v1",
        embedding_model_size: int = 1536,
        vector_store_path: str = "faiss_db/",
    ):
        super().__init__(embedding_model, embedding_model_size, vector_store_path)

    def create_vector_db(self):
        """Create an empty FAISS vector store.

        Args:
            config_path (str): Path to the chain's configuration file.
            vector_store_path (str): Path to save the empty vector store.
        """

        from faiss import IndexFlatL2

        # Load Ollama embeddings with specified model from config
        embeddings = BedrockEmbeddings(model_id=self.embedding_model)
        embedding_dimension = self.embedding_model_size

        # Create an empty FAISS vector store initialized with the embeddings dimension
        index = IndexFlatL2(embedding_dimension)  # Using L2 distance for the index

        # Create a local file store for persistent document storage
        docstore = InMemoryDocstore()

        # Create the FAISS vector store with the empty index and document store
        vector_store = FAISS(
            embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={}
        )

        # Save the empty vector store locally
        vector_store.save_local(folder_path=self.vector_store_path)
        return vector_store

    def delete_vector_db(self):
        # Delete the FAISS vector store
        try:
            os.remove(self.vector_store_path)
        except FileNotFoundError:
            pass

    def _load_pdfs_from_directory(self, directory_path: str):
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, "rb") as file:
                    pdf_reader = PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text:
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={"source": filename, "page": page_num + 1},
                                )
                            )
        return documents

    # %% Main pipeline function for ingesting and updating the vector database
    def ingest_documents(self, document_path) -> None:
        # Load documents from the specified PDF directory
        documents = self._load_pdfs_from_directory(document_path)

        # Load Ollama embeddings
        embeddings = BedrockEmbeddings(model_id=self.embedding_model)

        if documents:
            # Load the existing FAISS vector store
            vector_store = FAISS.load_local(
                folder_path=self.vector_store_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

            # Add documents to the vector store
            vector_store.add_documents(documents)

            # Save the updated vector store locally
            vector_store.save_local(folder_path=self.vector_store_path)


class QdrantVectorDB:
    def __init__(
        self,
        collection_name: str,
        embeddings_model: BedrockEmbeddings,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
    ):
        """
        Initialize Qdrant vector database and embeddings

        :param collection_name: Name of the Qdrant collection
        :param embeddings_model: Langchain embeddings model
        """
        # Initialize Qdrant client (in-memory for this example)
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embeddings = embeddings_model
        self.vector_size = vector_size
        self.distance = distance

    def create_vector_db(self):
        """
        Create a new collection in the Qdrant database

        :param collection_name: Name of the collection
        :param vector_size: Size of the vector
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def ingest_documents(self, folder_path: str, chunk_size=500, chunk_overlap=50) -> List[dict]:
        """
        Load documents from a specified folder

        :param folder_path: Path to the folder containing documents
        :return: List of processed documents
        """
        # Load documents from directory
        loader = DirectoryLoader(folder_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = text_splitter.split_documents(documents)

        # Generate embeddings and prepare for Qdrant
        points = []
        for idx, doc in enumerate(split_docs):
            # Generate embedding
            embedding = self.embeddings.embed_query(doc.page_content)

            points.append(
                {
                    "id": idx,
                    "vector": embedding,
                    "payload": {"page_content": doc.page_content, "metadata": doc.metadata},
                }
            )

        # Upsert points into Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        return points

    def load_documents_via_langchain(self, folder_path) -> QdrantVectorStore:
        """
        Load documents from a specified folder

        :param folder_path: Path to the folder containing documents
        :return: List of processed documents
        """

        # Load documents from directory
        loader = DirectoryLoader(folder_path)
        documents = loader.load()

        doc_store = QdrantVectorStore.from_documents(
            documents,
            self.embeddings,
            url="http://localhost:6333",
            collection_name=self.collection_name,
        )
        return doc_store

    def query_database(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Query the vector database

        :param query: Search query string
        :param top_k: Number of top results to return
        :return: List of top matching documents
        """
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query)

        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=top_k
        )

        if not search_result:
            return [{"score": 0.0, "text": "", "source": ""}]

        return [
            {
                "score": result.score,
                "text": result.payload.get("text", "") if result.payload else "",
                "source": result.payload.get("source", "") if result.payload else "",
            }
            for result in search_result
        ]
