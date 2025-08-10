
from uuid import uuid4
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

load_dotenv()
CHUNK_SIZE = 1000
EMBEDDING_DIR = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent/"resources/vectorstore"
COLLECTION_NAME = "real_estate"
llm = None
vector_store = None

def initialize_components():
  global llm, vector_store
  if not llm:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=512)
  
  if not vector_store:
    ef = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"trust_remote_code": True}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR)
    )


def process_urls(urls):
  yield "Initiale components"
  initialize_components()

  vector_store.reset_collection()

  yield "Loading urls"
  loader = UnstructuredURLLoader(urls)
  data =  loader.load()

  yield "Split text"
  text_splitter = RecursiveCharacterTextSplitter(
      separators=["\n\n", "\n", ".", " "],
      chunk_size=CHUNK_SIZE,
      # chunk_overlap=200
  )
  docs = text_splitter.split_documents(data)

  yield "Add docs to Vector DB"
  uuids = [str(uuid4()) for _ in range(len(docs))]
  vector_store.add_documents(docs, ids=uuids)

  # vector_store.persist()
  # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def generate_answer(query):
  if not vector_store:
    raise Exception("Vector store not initialized")

  chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
  result = chain.invoke({"question": query}, return_only_outputs=True)
  result.keys()
  return result["answer"], result["sources"]

if __name__ == "__main__":
  urls = [
    "https://www.pwc.com/us/en/industries/financial-services/asset-wealth-management/real-estate/emerging-trends-in-real-estate.html",
    "https://kpmg.com/in/en/blogs/2025/01/real-estate-2025-what-are-the-top-five-trends-to-watch-out-for.html"
  ]
  for msg in process_urls(urls):
    print(msg)  # Ensure the generator runs and you get the updates
  answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
  print(f"Answer: {answer}")
  print(f"Sources: {sources}")


# if __name__ == "__main__":
#   urls = [
#       "https://www.pwc.com/us/en/industries/financial-services/asset-wealth-management/real-estate/emerging-trends-in-real-estate.html",
#       "https://kpmg.com/in/en/blogs/2025/01/real-estate-2025-what-are-the-top-five-trends-to-watch-out-for.html"
#   ]

#   process_urls(urls)
#   answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")

#   print(f"Answer: {answer}")
#   print(f"Sources: {sources}")

  # results = vector_store.similarity_search(
  #   "30 year mortage rate",
  #   k=2,
  #   # filter={"source": "https://kpmg.com/in/en/blogs/2025/01/real-estate-2025-what-are-the-top-five-trends-to-watch-out-for.html"} 
  # )

  # for result in results:
  #   print(result.page_content)

