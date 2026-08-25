# NVIDIA Expert Assistant (RAG Project)

This repository contains a Retrieval-Augmented Generation (RAG) assistant built to answer domain-specific questions from a structured knowledge base (NVIDIA filings and related documents). The system is engineered for traceability, robust retrieval, and a streaming conversational UX.

## Project overview

- Frontend: Gradio chat UI that streams incremental responses and shows retrieved context.
- RAG orchestration: Query rewriting, vector retrieval, LLM-based reranking, prompt assembly, and streaming generation.
- Offline ingestion: Deterministic chunking, per-chunk contextualization, and embedding creation with stable ids.
- Vector store: Chroma (PersistentClient) storing embeddings and metadata for retrieval.

## Key files

- [app.py](</Users/kirtikarawat/RAG project/app.py>) — Gradio UI and streaming chat glue.
- [implement/answer.py](</Users/kirtikarawat/RAG project/implement/answer.py>) — RAG orchestration: rewrite_query, fetch_context, rerank, and streaming answer generator.
- [implement/ingest.py](</Users/kirtikarawat/RAG project/implement/ingest.py>) — Deterministic chunker, document summarization, chunk-context generation, and embedding insertion into Chroma.
- [pyproject.toml](</Users/kirtikarawat/RAG project/pyproject.toml>) — Project metadata and dependencies.
- [preprocessed_db/](</Users/kirtikarawat/RAG project/preprocessed_db>) — Chroma database files created by the ingestion pipeline.
- [data/](</Users/kirtikarawat/RAG project/data>) — Source documents (organized by fiscal year). Populate this with .md files for ingestion.

## Architecture (brief)

User (Browser) → Gradio UI (app.py) → RAG controller (implement/answer.py) →
Chroma vector DB (preprocessed_db) + LLM service (litellm / Ollama) + Embedding model (SentenceTransformers)

For a visual diagram, see the repository's Mermaid/diagram file (if present) or render the Mermaid source provided separately.

## Quickstart (local development)

1. Create a Python 3.12+ virtual environment and activate it:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)

2. Install dependencies

   The project lists dependencies in `pyproject.toml` (e.g. chromadb, gradio, litellm, sentence-transformers).

   You can install the main packages with pip (example):

   python -m pip install -U pip
   python -m pip install chromadb gradio litellm sentence-transformers python-dotenv pydantic tenacity tqdm

   (Or use your preferred environment tool: poetry, pip-tools, etc.)

3. Prepare environment variables

   - Copy `.env.example` (if present) to `.env` and add any required keys (model endpoints, API keys, etc.).
   - The code expects a local LLM endpoint compatible with `litellm` (the default MODEL in the code is `ollama/llama3.2`).

4. Ingest documents (create/update the vector DB)

   - Place .md files under `data/<year>/` (for example `data/2025/*.md`).
   - Run the ingestion script:

     python implement/ingest.py

   This will create the Chroma collection and populate `preprocessed_db/` with the vector store files.

5. Run the Gradio app

   python app.py

   The app launches in-browser by default. Ask questions and the system will stream answers with retrieved context.

## Engineering notes & best practices

- Deterministic chunking with overlap and stable ids (source::chunk_index) enables incremental re-ingestion and safer upserts.
- One document-level summary call per document reduces LLM usage when generating chunk-level context.
- Query rewriting + rerank helps improve retrieval precision.
- Lazy-loading of embedding models avoids duplicated memory usage across worker processes.
- The system is designed for local development with a local LLM (Ollama). For production, consider containerization, secure endpoints for LLMs, and an external managed vector DB for scale.

## Deployment suggestions

- Containerize the app, the ingestion worker, and the LLM service (if allowed).
- Use volumes for persistent Chroma storage and schedule regular backups.
- Protect the Gradio endpoint with authentication before exposing publicly.

## Contributing

Contributions and suggestions are welcome. Please open issues or PRs for improvements (ingest performance, streaming UX, security hardening).

## License

Add your license here.

