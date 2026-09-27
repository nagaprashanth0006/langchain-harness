import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from vector_store_operations import ensure_vectorstore, retrieve_context

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MODEL_CONTEXT = 4092


def create_chatbot():
    return ChatOllama(
        model=LLM_MODEL,
        num_ctx=MODEL_CONTEXT,
        temperature=0.3,
        base_url=OLLAMA_BASE_URL,
    )


def direct_response(prompt, context):
    messages = [
        (
            "system",
            "Answer the question using the supplied context. Be concise. If the context does not contain the answer, say so.",
        ),
        ("human", f"Context:\n{context}\n\nQuestion: {prompt}"),
    ]
    return create_chatbot().invoke(messages).content


def rag_response(prompt, collection_name="documentation"):
    store = ensure_vectorstore(collection_name)
    context = retrieve_context(store, prompt, k=3)
    return direct_response(prompt, context)


if __name__ == "__main__":
    print(rag_response("What does the documentation cover?"))
