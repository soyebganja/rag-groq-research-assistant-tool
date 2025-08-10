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

📦 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/<your-username>/research-assistance-tool.git
cd research-assistance-tool
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
⚙️ Environment Variables
Create a .env file in the root directory and add:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
▶️ Usage
Run the Application
bash
Copy
Edit
python app.py
Example Python Usage
python
Copy
Edit
from rag import ingest_pdf, ingest_url, answer_query

# Ingest PDF
ingest_pdf("sample.pdf")

# Ingest URL
ingest_url("https://example.com/article")

# Ask a question
response = answer_query("What is the main topic of the article?")
print(response)
📂 Project Structure
bash
Copy
Edit
research-assistance-tool/
│
├── app.py             # Flask API entry point
├── rag.py             # Core logic for ingestion & querying
├── requirements.txt   # Python dependencies
├── .env               # API keys (not tracked in git)
└── README.md          # Project documentation
📜 API Endpoints
Method	Endpoint	Description
POST	/ingest-pdf	Upload a PDF and store embeddings
POST	/ingest-url	Ingest website content
POST	/query	Ask a question to the knowledge DB

🧠 How It Works
Ingestion – The system extracts text from PDFs or URLs.

Embedding – Text is converted into vector embeddings using LangChain.

Storage – FAISS stores these embeddings locally.

Querying – User queries are embedded and compared with stored vectors.

Answering – Groq LLM generates a contextual answer from the matched text.

🔮 Future Improvements
Add support for DOCX ingestion

Deploy as a web app with file upload & chat UI

Add multi-language support

Integration with Google Drive and Notion

📄 License
This project is licensed under the MIT License – feel free to use and modify it.
