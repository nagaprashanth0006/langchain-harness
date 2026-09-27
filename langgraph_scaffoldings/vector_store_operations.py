"Datastores:"

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from os import path, mkdir, makedirs, listdir
from langchain_core.documents import Document

import hashlib
import fitz
import os
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PERSISTENCE_DIR = "./vector-store"
DATA_DIR = "./data"
EMBEDDINGS_MODEL = "nomic-embed-text"

if not path.exists(PERSISTENCE_DIR):
    mkdir(PERSISTENCE_DIR)
if not path.exists(DATA_DIR):
    mkdir(DATA_DIR)


def sid(page_content):
    return hashlib.md5(page_content.encode("utf-8")).hexdigest()


def uploaded_file_to_text(uploaded_file):
    uploaded_file.seek(0)
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": uploaded_file.name})]


def uploaded_file_to_pdf(uploaded_file):
    uploaded_file.seek(0)
    blob = uploaded_file.read()
    pdf = fitz.open(stream=blob, filetype="pdf")
    out: list[Document] = []
    for i, page in enumerate(pdf):
        text = page.get_text() or ""
        if text.strip():
            out.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": i + 1,
                        "pages": pdf.page_count,
                    },
                )
            )
    pdf.close()
    return out


def _uploads_to_documents(uploaded_files) -> list[Document]:
    docs: list[Document] = []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower()
        if suffix == ".txt":
            docs.extend(uploaded_file_to_text(uf))
        elif suffix == ".pdf":
            docs.extend(uploaded_file_to_pdf(uf))
    print(
        f"Given {len(uploaded_files)} uploaded files to documents, returning {len(docs)} documents."
    )
    return docs


def ensure_vectorstore(collection_name):
    makedirs(PERSISTENCE_DIR, exist_ok=True)
    embeddings = OllamaEmbeddings(
        model=EMBEDDINGS_MODEL, num_ctx=8192, base_url=OLLAMA_BASE_URL
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=PERSISTENCE_DIR,
        embedding_function=embeddings,
    )


def vectorize_docs(uploaded_files, collection_name="default"):
    vs = ensure_vectorstore(collection_name=collection_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100, length_function=len
    )

    try:
        docs = _uploads_to_documents(uploaded_files)
        if not docs:
            return False, "No text extracted."
        chunks = splitter.split_documents(docs)
        unique = {sid(chunk.page_content): chunk for chunk in chunks}
        vs.add_documents(list(unique.values()), ids=list(unique))
        return True, ""
    except Exception as e:
        print("Exception while vectorizing:", e)
        return False, e


def refresh_embeddings(collection_name="default"):
    vs = ensure_vectorstore(collection_name=collection_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, length_function=len
    )
    text_files = [
        path.join(DATA_DIR, f) for f in listdir(DATA_DIR) if f.lower().endswith(".txt")
    ]
    for file in text_files:
        docs = TextLoader(file, encoding="utf-8").load()
        chunks = splitter.split_documents(docs)
        ids = [sid(c.page_content) for c in chunks]
        vs.add_documents(chunks, ids=ids)

    pdf_files = [
        path.join(DATA_DIR, f) for f in listdir(DATA_DIR) if f.lower().endswith(".pdf")
    ]
    for file in pdf_files:
        docs = PyMuPDFLoader(file).load()
        chunks = splitter.split_documents(docs)
        ids = [sid(c.page_content) for c in chunks]
        vs.add_documents(chunks, ids=ids)


def retrieve_context(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)


def check_vs(collection_name):
    vs = ensure_vectorstore(collection_name)
    col_name = getattr(vs, "_collection")
    col_count = vs._collection.count()
    print("Collection name", col_name, "Count of documents:", col_count)
    return (col_name, col_count)


if __name__ == "__main__":
    collections_name = "documentation"
    vs = ensure_vectorstore(collections_name)
    print(
        "Collection name",
        getattr(vs, "_collection"),
        "Count of documents:",
        vs._collection.count(),
    )
    refresh_embeddings(collection_name=collections_name)
    print("After refresh")
    print(
        "Collection name",
        getattr(vs, "_collection"),
        "Count of documents:",
        vs._collection.count(),
    )

    query = "What is name of frontend project"
    context = retrieve_context(vs, query)
    print(context)
