"""
Chat with My CV — a small Retrieval-Augmented Generation (RAG) demo.

How it works (the RAG pattern):
1. Load the CV text and split it into overlapping chunks.
2. Embed each chunk into a vector.
   - Primary path: ChromaDB's built-in local embedding model
     (all-MiniLM-L6-v2, a real sentence-embedding transformer model,
     running fully offline via ONNX once its ~90MB weights are cached —
     no API key, no per-query cost).
   - Fallback path: if that model can't be downloaded (e.g. no internet
     access, or a network that blocks the model host), the app
     automatically falls back to a TF-IDF vectorizer from scikit-learn,
     so the whole pipeline still runs end to end offline with zero
     external dependencies.
3. Store the vectors in a local ChromaDB vector database.
4. When a question comes in, embed the question the same way and run a
   similarity search to retrieve the most relevant chunk(s) — this is the
   "Retrieval" step.
5. Return the retrieved chunk(s) as the grounded answer — this is the
   simplest possible "Generation" step: instead of calling a paid LLM API,
   we surface the exact source text the answer came from, so every answer
   is verifiable and hallucination-free.

Run:
    pip install chromadb scikit-learn
    python app.py
"""

import re
import sys

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


def load_and_chunk(path: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Read a text file and split it into overlapping chunks.

    Overlap keeps a sentence that spans a chunk boundary from being cut
    in half and losing meaning for the retriever.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return [c for c in chunks if c.strip()]


class TfidfEmbeddingFunction(EmbeddingFunction):
    """Offline fallback embedding function.

    Used automatically when the semantic MiniLM model can't be downloaded
    (e.g. no internet access). It represents each chunk as a TF-IDF vector
    fitted on the CV's own vocabulary — a classic, fully local vector
    representation with no external model weights required.
    """

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer().fit(corpus)

    def __call__(self, input: Documents) -> Embeddings:
        return self.vectorizer.transform(input).toarray().tolist()


def get_embedding_function(corpus: list[str]):
    """Try the real semantic embedding model; fall back to TF-IDF offline."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    try:
        ef = DefaultEmbeddingFunction()
        ef(["connectivity check"])  # triggers the model download on first use
        print("Using semantic embeddings (all-MiniLM-L6-v2, via ONNX, local).")
        return ef
    except Exception as e:
        print(f"Semantic embedding model unavailable ({type(e).__name__}: {e}).")
        print("Falling back to a local TF-IDF vectorizer (fully offline, no download).")
        return TfidfEmbeddingFunction(corpus)


def build_vector_store(chunks: list[str], collection_name: str = "cv_chunks"):
    """Embed the chunks and store them in a local, on-disk ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")

    # Recreate the collection fresh each run so re-running app.py never
    # duplicates old chunks.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_function = get_embedding_function(chunks)

    collection = client.create_collection(
        collection_name, embedding_function=embedding_function
    )

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def answer_question(collection, question: str, top_k: int = 2) -> None:
    """Retrieve the most relevant chunk(s) for a question and print them."""
    results = collection.query(query_texts=[question], n_results=top_k)

    documents = results["documents"][0]
    distances = results["distances"][0]

    print(f"\nQ: {question}")
    for rank, (doc, dist) in enumerate(zip(documents, distances), start=1):
        similarity = round(1 - dist, 3)
        print(f"\n  [{rank}] (similarity score: {similarity})")
        print(f"  {doc.strip()}")


def main():
    print("Loading and chunking CV...")
    chunks = load_and_chunk("data/cv.txt")
    print(f"Created {len(chunks)} chunks.\n")

    collection = build_vector_store(chunks)

    sample_questions = [
        "What Business Intelligence experience does this candidate have?",
        "What programming languages does this candidate know?",
        "What projects has this candidate built?",
        "What is this candidate's education background?",
    ]

    for q in sample_questions:
        answer_question(collection, q)

    if len(sys.argv) > 1 and sys.argv[1] == "--demo-only":
        return

    print("\n\nDone. Try your own question below (or press Enter to quit):")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except EOFError:
            break
        if not q:
            break
        answer_question(collection, q)


if __name__ == "__main__":
    main()
