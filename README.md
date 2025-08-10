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

### 2️⃣ Create a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

### 3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

### 4️⃣ Add your environment variables
Create a .env file in the root directory:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
▶️ Usage
Start the Streamlit app
bash
Copy
Edit
streamlit run app.py
Ingest a URL
Paste the article/research paper URL into the input box

Click Ingest

Data is processed, chunked, and stored in FAISS

Ingest a PDF
Upload your PDF

System extracts text, chunks it, and stores vectors

Ask Questions
Type your query in the input box

The system retrieves the most relevant chunks

GROQ LLM generates a precise answer

### 📸 Example
Question:

What are the main findings from the latest AI research in the uploaded paper?

Answer:

The study concludes that integrating multimodal models with self-corrective feedback significantly improves reasoning accuracy, particularly in open-ended problem-solving scenarios.

### 🧭 Project Structure
bash
Copy
Edit
research-assistance-tool/
│
├── app.py           # Streamlit UI
├── rag.py           # Ingestion, storage, retrieval, Q&A
├── requirements.txt # Dependencies
├── .env.example     # Example env file
└── README.md        # Documentation

### 🔮 Future Enhancements
Multi-file batch ingestion

Support for audio/video transcription

Integration with other LLMs (Claude, Gemini)

User authentication & role-based access

Cloud-hosted vector store

###  📜 License
This project is licensed under the MIT License.

### 💡 Acknowledgements
LangChain

FAISS

GROQ

Streamlit

yaml
Copy
Edit

---

If you want, I can also **add real usage screenshots and architecture diagram** to make your GitHub repo stand out.  
I can prepare those next so your README looks professional.
