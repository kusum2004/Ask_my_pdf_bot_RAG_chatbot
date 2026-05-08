from typing import List, Any, Optional

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config


class EmbeddingManager:
    """
    Manages embeddings and retrieval using SentenceTransformer embeddings
    and FAISS vector storage.
    """

    def __init__(self):
        self.embedding_model: Optional[Any] = None
        self.vectorstore = None
        self.retriever = None

    def _init_embedding_model(self):
        """
        Lazily initialize the embedding model.
        """

        if self.embedding_model is not None:
            return

        try:
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )

            print("SentenceTransformer Embeddings initialized successfully.")

        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            raise e

    def create_embeddings(self, documents: List[Document]):
        """
        Creates embeddings for documents and stores them in FAISS.

        Args:
            documents: List of LangChain Document objects
        """

        try:
            if not documents:
                print("No documents provided.")
                return False

            # Initialize embedding model
            self._init_embedding_model()

            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": Config.TOP_K}
            )

            print("Embeddings created successfully.")

            return True

        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            return False

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Searches for relevant documents.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant documents
        """

        if not k:
            k = Config.TOP_K

        if not self.vectorstore:
            return []

        try:
            if not self.retriever and self.vectorstore:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )

            # Updated retrieval method
            relevant_docs = self.retriever.invoke(query)

            return relevant_docs

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def clear_embeddings(self):
        """
        Clears vectorstore and retriever.
        """

        self.vectorstore = None
        self.retriever = None