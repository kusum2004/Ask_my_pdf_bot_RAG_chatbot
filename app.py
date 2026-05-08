import streamlit as st
import time
from src.processor import PDFProcessor
from src.embedding import EmbeddingManager
from src.chat import ChatManager
from src.config import Config
import json
from langchain_core.documents import Document

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

def generate_txt(answer: str):
    return answer.encode("utf-8")

def generate_pdf(answer: str):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(answer, styles["Normal"])]
    doc.build(story)
    buffer.seek(0)
    return buffer

def initialize_session_state():
    if "processor" not in st.session_state:
        st.session_state.processor = PDFProcessor()
        
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
        
    if "chat_manager" not in st.session_state:
        if not Config.is_valid():
            st.error("Missing API key. Please check your .env file.")
            return False
        st.session_state.chat_manager = ChatManager(Config.GOOGLE_API_KEY)
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "documents" not in st.session_state:
        st.session_state.documents = []
        
    return True

def process_documents(uploaded_files):
    try:
        with st.spinner("Processing documents..."):
            all_documents = []
            
            for file in uploaded_files:
                documents = st.session_state.processor.process_document(file)
                all_documents.extend(documents)
                
            st.session_state.documents = all_documents

            if not all_documents:
                st.warning("No text extracted.")
                return False

            success = st.session_state.embedding_manager.create_embeddings(all_documents)
            
            if success:
                st.session_state.chat_manager.set_retriever(
                    st.session_state.embedding_manager.retriever
                )
                st.success(f"{len(all_documents)} chunks created!")
                return True
            else:
                st.error("Embedding failed.")
                return False
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def main():
    if not initialize_session_state():
        return
    
    st.title("📚 PDF Chat Assistant")
    
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write("Uploaded Files:")
            for file in uploaded_files:
                st.write(f"- {file.name}")

        if uploaded_files:
            if st.button("Process Documents"):
                process_documents(uploaded_files)

        if st.session_state.documents:
            pdf_names = list(set([doc.metadata.get("source") for doc in st.session_state.documents]))
            selected_pdf = st.selectbox("Select PDF", ["All"] + pdf_names)
        else:
            selected_pdf = "All"

        if st.session_state.documents:
            st.success(f"{len(st.session_state.documents)} chunks loaded")
            
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.session_state.chat_manager.reset_conversation()
                st.rerun()
            
            if st.button("Clear File Chunks"):
                st.session_state.documents = []
                st.session_state.embedding_manager.clear_embeddings()
                st.rerun()

    #  CHAT HISTORY
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #  INPUT
    if query := st.chat_input("Ask your question"):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)

        if not st.session_state.documents:
            st.warning("Upload PDFs first")
            return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                all_docs = st.session_state.embedding_manager.search(query)

                if selected_pdf != "All":
                    relevant_docs = [
                        doc for doc in all_docs
                        if doc.metadata.get("source") == selected_pdf
                    ]
                else:
                    relevant_docs = all_docs

                response = st.session_state.chat_manager.generate_response(query, relevant_docs)

                if isinstance(response, dict):
                    answer = response.get("answer", "")
                else:
                    answer = str(response)

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="Download as TXT",
                        data=generate_txt(answer),
                        file_name="response.txt",
                        mime="text/plain"
                    )

                with col2:
                    st.download_button(
                        label="Download as PDF",
                        data=generate_pdf(answer),
                        file_name="response.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()