# Research Assistance Tool (RAT)

A powerful **AI-powered Research Assistance Tool** that lets you **ingest URLs and PDFs, store them in a vector database, and query them using natural language**.  
This tool uses **GROQ LLM** with **LangChain** to provide **fast, accurate, and context-aware responses**.

<img width="1284" height="690" alt="image" src="https://github.com/user-attachments/assets/0bbb0508-5942-447b-bdb7-9b2f10d89105" />

---

## 🚀 Features

- **URL & PDF ingestion**  
  Upload research papers, articles, and documents to build your knowledge base.
  
- **Vector Store for Search**  
  Uses **FAISS** for fast semantic search.
  
- **GROQ LLM-powered Q&A**  
  Ask natural language questions and get precise, context-rich answers.
  
- **Web UI via Streamlit**  
  Simple, interactive interface for ingestion and querying.
  
- **Chunked Data Processing**  
  Splits documents into meaningful chunks for better retrieval.

---

## 🛠 Tech Stack

- **[Python 3.10+]**
- **[LangChain](https://www.langchain.com/)** – Orchestrating ingestion, chunking, and retrieval  
- **[FAISS](https://github.com/facebookresearch/faiss)** – Efficient vector similarity search  
- **[GROQ LLM](https://groq.com/)** – High-speed inference with Llama3 models  
- **[Streamlit](https://streamlit.io/)** – User interface for easy interaction  
- **[PyPDF2](https://pypi.org/project/pypdf2/)** – PDF parsing and text extraction  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/research-assistance-tool.git
cd research-assistance-tool
