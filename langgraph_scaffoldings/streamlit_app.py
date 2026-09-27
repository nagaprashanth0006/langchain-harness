import streamlit as st
from rag_chat import rag_response
from vector_store_operations import check_vs, vectorize_docs

st.title("Document chat")
collection = "document_chat"
try:
    _, count = check_vs(collection)
    st.caption(f"Indexed chunks: {count}")
except Exception as exc:
    st.error(f"Could not open the document store: {exc}")
    st.stop()

uploads = st.file_uploader("Documents", type=["txt", "pdf"], accept_multiple_files=True)
if st.button("Index documents", disabled=not uploads):
    try:
        with st.spinner("Indexing documents"):
            success, error = vectorize_docs(uploads, collection_name=collection)
        if success:
            st.success("Documents indexed. You can ask a question now.")
        else:
            st.error(f"Could not index documents: {error}")
    except Exception as exc:
        st.error(f"Could not index documents: {exc}")

prompt = st.text_input("Question", placeholder="What does the document explain?")
if st.button("Ask", disabled=not prompt.strip()):
    try:
        with st.spinner("Reading the documents"):
            st.write(rag_response(prompt, collection_name=collection))
    except Exception as exc:
        st.error(f"Could not answer. Check Ollama and the selected models: {exc}")
