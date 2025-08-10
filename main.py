import streamlit as st
from rag import process_urls, ask

st.set_page_config(page_title="Research Assistance Tool", layout="wide")

st.title("📚 Research Assistance Tool")

# Default URLs
urls = [
    "https://www.pwc.com/us/en/industries/financial-services/asset-wealth-management/real-estate/emerging-trends-in-real-estate.html",
    "https://kpmg.com/in/en/blogs/2025/01/real-estate-2025-what-are-the-top-five-trends-to-watch-out-for.html"
]

# URL processing section
st.subheader("1️⃣ Load & Process URLs")
if st.button("Process URLs"):
    with st.spinner("Processing URLs..."):
        process_urls(urls)
    st.success("✅ URLs processed and stored in vector database.")

# Query section
st.subheader("2️⃣ Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            ask(query)
