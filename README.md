
# 📄 Ask My PDF Bot - RAG Chatbot

An AI-powered PDF Chat Assistant built using Streamlit, LangChain, FAISS, and Google Gemini API.  
This application allows users to upload PDF documents and ask questions directly from the uploaded content using Retrieval-Augmented Generation (RAG).

---

# 🚀 Features

- Upload and process PDF documents
- Extract and chunk PDF text
- Semantic search using embeddings
- AI-powered question answering
- Retrieval-Augmented Generation (RAG)
- Interactive Streamlit UI
- Google Gemini integration
- FAISS vector database for fast retrieval

---

# 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS
- Sentence Transformers
- PyMuPDF
- HuggingFace Embeddings

---

# 📂 Project Structure

```bash
Ask-My-PDF-Bot-RAG-Chatbot/
│
├── app.py
├── requirements.txt
├── .env
│
├── data/
│
├── src/
│   ├── chat.py
│   ├── config.py
│   ├── embedding.py
│   ├── processor.py
│
└── README.md
````

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/Ask-My-PDF-Bot-RAG-Chatbot.git
```

## 2️⃣ Navigate to Project

```bash
cd Ask-My-PDF-Bot-RAG-Chatbot
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure API Key

Create a `.env` file in the root directory and add:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

Get API Key from:

[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🧠 How It Works

1. User uploads PDF files
2. PDF text is extracted and chunked
3. Embeddings are generated using Sentence Transformers
4. FAISS stores vector embeddings
5. User asks questions
6. Relevant chunks are retrieved
7. Gemini generates contextual answers

---

# 📸 Application Features

* PDF Upload Interface
* AI Chat Assistant
* Context-Aware Responses
* Real-time Processing
* Download Responses

---

# 🔮 Future Improvements

* Multi-PDF support
* Chat history memory
* Voice interaction
* OCR support for scanned PDFs
* Deployment on cloud platforms

---

# 👨‍💻 Author

Developed as an AI & Machine Learning project using RAG architecture and Generative AI technologies.


