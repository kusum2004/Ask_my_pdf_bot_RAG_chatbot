from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from src.config import Config


class ChatManager:

    def __init__(self, api_key: str):

        self.api_key = api_key

        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.3,
            max_output_tokens=1024
        )

    def generate_response(self, query: str, context_docs: List[Document]):

        try:

            # Extract text from retrieved documents
            context_text = "\n\n".join(
                [doc.page_content for doc in context_docs]
            )

            # Prompt
            prompt = f"""
You are a helpful PDF assistant.

Answer the question ONLY using the provided context.

If the answer is not available in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context_text}

Question:
{query}
"""

            # Generate response
            response = self.llm.invoke(prompt)

            return {
                "answer": response.content,
                "sources": []
            }

        except Exception as e:

            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }

    def set_retriever(self, retriever):
        pass

    def reset_conversation(self):
        pass

    def get_conversation_history(self):
        return []