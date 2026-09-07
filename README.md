# Chat with My CV — a RAG Demo

A small Retrieval-Augmented Generation (RAG) project that lets you ask
natural-language questions about a CV and get back the exact, grounded
passage that answers them — instead of a language model guessing or
hallucinating an answer.

## Why this exists

This project is a hands-on demonstration of the RAG pattern: chunking a
document, embedding it into a vector space, storing it in a vector
database, and retrieving the most relevant piece of text for a question
through semantic similarity search.

## How it works

1. **Load & chunk** — the CV text is split into overlapping ~500-character
   chunks (with a 100-character overlap so no sentence gets cut in half
   and loses meaning).
2. **Embed** — each chunk is converted into a numeric vector.
   - Primary: ChromaDB's built-in `all-MiniLM-L6-v2` sentence-embedding
     model, running locally via ONNX (no API key, no per-query cost).
   - Automatic offline fallback: if that model's weights can't be
     downloaded (no internet, or a restricted network), the app switches
     to a TF-IDF vectorizer (scikit-learn) fitted on the CV's own
     vocabulary, so the whole pipeline still works with zero external
     calls.
3. **Store** — vectors are stored in a local, on-disk ChromaDB vector
   database (no cloud service required).
4. **Retrieve** — a question is embedded the same way, and a similarity
   search finds the most relevant chunk(s) in the vector database.
5. **Answer** — the retrieved chunk is returned directly as the answer.
   This keeps every answer verifiable and grounded in the source
   document — there's no generation step that could hallucinate, since
   the "answer" is always a real excerpt from the CV.

## Tech stack

- Python 3
- [ChromaDB](https://www.trychroma.com/) — vector database
- [scikit-learn](https://scikit-learn.org/) — TF-IDF fallback vectorizer

## Run it

```bash
pip install -r requirements.txt
python app.py
```

The script runs four sample questions automatically, then lets you type
your own questions interactively.

## Example

```
Q: What Business Intelligence experience does this candidate have?

  [1] (similarity score: 0.71)
  Business Intelligence Trainee - Concentrix - Amman, Jordan - May 2023
  to Oct 2023: Designed and maintained interactive Power BI dashboards
  and performance reports used directly by team leads to guide business
  decisions...
```

## Possible next steps

- Swap the extractive "return the chunk" step for a real generation step
  by piping the retrieved chunk(s) into an LLM API (OpenAI, Anthropic,
  or a local model) to produce a natural-language answer grounded in the
  retrieved context.
- Swap the CV for any other document (notes, a report, a book chapter)
  by replacing `data/cv.txt`.
- Add a simple web UI (Streamlit or Flask) on top of `answer_question()`.
