# Research Assistance Tool (RAT)

A powerful **AI-powered Research Assistance Tool** that lets you **ingest URLs and PDFs, store them in a vector database, and query them using natural language**.  
This tool uses **GROQ LLM** with **LangChain** to provide **fast, accurate, and context-aware responses**.

**Live Demo Link**: [RAG-GROQ-Research-Assistant-Tool](https://rag-groq-research-assistant-tool-hvcyd5qpjwhsa2uki8yyo9.streamlit.app/)
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

## 📂 Project Structure

```
groq-rag-app/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed to Git)
├── README.md              # Project documentation
├── data/                  # Folder for uploaded documents
└── utils.py
```

---

## 🛠️ Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/<your-username>/groq-rag-app.git
cd groq-rag-app
```

2️⃣ **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Set up environment variables**

Create a `.env` file in the root directory and add:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Once running, open your browser at:
```
http://localhost:8501
```

---

## 📖 How It Works

1. **Load Documents** – PDF/Text files are loaded and split into chunks.
2. **Embed Documents** – GROQ-compatible embeddings are generated and stored in a vector DB.
3. **Retrieve Context** – Based on your query, relevant document chunks are retrieved.
4. **Generate Answer** – GROQ LLM generates a contextual, accurate response.

---

## 📦 Dependencies

- **Python** 3.9+
- **LangChain**
- **GROQ API**
- **FAISS / Chroma**
- **Streamlit**
- **dotenv**

Install them with:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch (`feature/my-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 📧 Contact

**Author:** Soyeb Ganja 
**Email:** soyeb.ganja@gmail.com
**Soyeb Ganja** - [LinkedIN Profile](https://linkedin.com/in/soyeb-ganja), [GitHub Profile](https://github.com/soyebganja)


