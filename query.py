import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Embeddings load karo
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Vector store load karo
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 3. Retriever banao
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

while True:
    query = input("\n❓ Apna sawal likho (ya 'exit'): ")
    if query.lower() == "exit":
        break

    # Similar docs retrieve karo
    docs = retriever.get_relevant_documents(query)

    print("\n📄 Relevant Passages:")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Passage {i} ---\n{doc.page_content}")
