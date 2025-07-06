from sentence_transformers import SentenceTransformer

texts = ["Embeddings represent text as vectors.", "Python is a programming language."]

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    print(embeddings.shape)
    print(embeddings)
