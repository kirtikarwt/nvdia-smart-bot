"""

Key improvements over the previous version:
  1. Deterministic chunking (fixed size + overlap) instead of trusting an LLM
     to split the document AND guarantee full coverage. Coverage is now
     guaranteed by code.
  2. "Contextual Retrieval" pattern: for each chunk, an LLM is asked to
     produce a short blurb situating that chunk within the whole document
     (using the *whole document* as context). That blurb is prepended to the
     chunk before embedding, which meaningfully improves retrieval quality
     for chunks that are ambiguous in isolation (e.g. "Revenue grew 12%"
     becomes "This chunk is from NVIDIA's fiscal year 2024 10-K MD&A
     section...").
  3. Embedding model choice/chunk size is aligned with the model's actual
     max sequence length so text isn't silently truncated during embedding.
  4. Embedding model is lazy-loaded inside the process that uses it, so
     multiprocessing workers (which only do LLM calls) don't each load a
     redundant copy into memory.
  5. Richer metadata (chunk_index, doc_chunk_count, headline, fiscal_year) so
     you can fetch neighboring chunks at query time for extra context, or
     filter by year without re-parsing page_content — important here since
     users frequently ask year-specific questions ("revenue in 2023" vs
     "2024") and the folder structure already encodes fiscal year.
  6. Retries are capped (stop_after_attempt) so one persistently bad
     document can't hang the whole ingestion run.
  7. A coverage check flags documents where chunking seems to have lost
     content, instead of silently trusting the split.
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv(override=True)

MODEL = "ollama/llama3.2"

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data"

# BAAI/bge-base-en-v1.5 has a 512 *token* max sequence length (vs MiniLM's
# 256), and consistently outperforms MiniLM on retrieval benchmarks. We still
# size chunks conservatively below the limit so headline + context + original
# text together don't get truncated. ~4 chars/token is a safe rule of thumb
# for English.
#
# NOTE: BGE models are trained asymmetrically. Passages/chunks are embedded
# as-is (no prefix) — only queries need an instruction prefix at retrieval
# time. See the query-side file for that half.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE_CHARS = 1400         # ~350 tokens of original_text
CHUNK_OVERLAP_CHARS = 250       # ~18% overlap, in line with the original spec

# By default, a local Ollama server processes one request at a time unless
# you've explicitly configured it for parallelism (OLLAMA_NUM_PARALLEL env
# var). If Ollama is only handling one request at a time, running multiple
# WORKERS here just means several worker processes queueing up behind the
# same single-threaded server — no real speedup, and possibly slower due to
# process overhead. If you haven't configured Ollama for parallel requests,
# WORKERS = 1 is often the more honest (and easier to reason about) setting.
WORKERS = 1
wait = wait_exponential(multiplier=1, min=10, max=240)


class Result(BaseModel):
    page_content: str
    metadata: dict


class ChunkContext(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, a few words, likely to be surfaced in a query"
    )
    context: str = Field(
        description=(
            "1-2 sentences that situate this chunk within the overall document "
            "(e.g. what document it's from, what section/topic it covers) so the "
            "chunk makes sense when read completely on its own, out of context."
        )
    )


class DocumentSummary(BaseModel):
    summary: str = Field(
        description=(
            "A concise summary (3-6 sentences) of the whole document: what it is, "
            "what fiscal year/topic it covers, and its main sections or subject matter."
        )
    )


def fetch_documents():
    """
    A homemade version of the LangChain DirectoryLoader.

    Top-level subfolders under data/ are fiscal years (2023, 2024, 2025, ...),
    each containing NVIDIA filing documents (10-K sections, proxy statements,
    shareholder letters, etc.) as .md files.
    """
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        fiscal_year = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {"fiscal_year": fiscal_year, "source": file.as_posix(), "text": f.read()}
                )

    print(f"Loaded {len(documents)} documents")
    return documents


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Deterministic recursive-ish splitter: tries to break on paragraph, then
    sentence, then whitespace boundaries so chunks don't cut mid-word, while
    *guaranteeing* full coverage of the input text (unlike relying on an LLM
    to return complete coverage).
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        sep = seps[0] if seps else ""
        parts = t.split(sep) if sep else list(t)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size and len(seps) > 1:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap by stitching the tail of each chunk onto the front of the next
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = raw_chunks[i - 1][-overlap:] if overlap else ""
            overlapped.append((prev_tail + chunk).strip())
    return [c for c in overlapped if c.strip()]


def make_summary_prompt(document: dict) -> str:
    return f"""You are helping build a knowledge base for a chatbot representing NVIDIA, the technology company.

Here is a document (source: {document["source"]}, fiscal year: {document["fiscal_year"]}):

<document>
{document["text"]}
</document>

Write a concise summary of this document: what it is, what fiscal year/topic it covers,
and what its main sections or subject matter are. This summary will be reused as context
for every chunk of this document, so keep it focused and information-dense rather than
long."""


@retry(wait=wait, stop=stop_after_attempt(5))
def get_document_summary(document: dict) -> str:
    """
    One LLM call per DOCUMENT (not per chunk) that produces a short summary.
    That summary is then reused as lightweight context for every chunk's
    headline/context generation, instead of resending the full document text
    on every single chunk call. For a document with N chunks, this turns N
    full-document-length LLM calls into 1 full-document-length call + N
    short-summary-length calls — a large speedup, especially for longer
    documents (10-K sections, proxy statements, etc.) on local/CPU-bound
    models.
    """
    messages = [{"role": "user", "content": make_summary_prompt(document)}]
    response = completion(model=MODEL, messages=messages, response_format=DocumentSummary)
    reply = response.choices[0].message.content
    return DocumentSummary.model_validate_json(reply).summary


def make_context_prompt(document: dict, doc_summary: str, chunk: str) -> str:
    return f"""You are helping build a knowledge base for a chatbot representing NVIDIA, the technology company.

Here is a summary of the full document this chunk comes from (source: {document["source"]}, fiscal year: {document["fiscal_year"]}):

<document_summary>
{doc_summary}
</document_summary>

Here is a specific chunk taken from that document:

<chunk>
{chunk}
</chunk>

Give this chunk a short headline and a 1-2 sentence piece of context that situates it
within the overall document, so someone reading only the chunk (without the rest of
the document) understands what it's about, which fiscal year it relates to, and where
it comes from. Do not repeat the full chunk text back."""


def make_messages(document, doc_summary, chunk):
    return [{"role": "user", "content": make_context_prompt(document, doc_summary, chunk)}]


@retry(wait=wait, stop=stop_after_attempt(5))
def get_chunk_context(document: dict, doc_summary: str, chunk: str) -> ChunkContext:
    messages = make_messages(document, doc_summary, chunk)
    response = completion(model=MODEL, messages=messages, response_format=ChunkContext)
    reply = response.choices[0].message.content
    return ChunkContext.model_validate_json(reply)


def process_document(document: dict) -> list[Result]:
    raw_chunks = split_text(document["text"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)

    # Coverage sanity check: flag (don't fail) documents where the chunker
    # seems to have lost a meaningful amount of content.
    covered_chars = sum(len(c) for c in raw_chunks)
    if covered_chars < 0.9 * len(document["text"]):
        print(
            f"WARNING: possible coverage loss for {document['source']} "
            f"({covered_chars} chars covered vs {len(document['text'])} original)"
        )

    print(f"[{document['source']}] {len(raw_chunks)} chunks — summarizing document...")

    # One summary call per document, reused for every chunk below — this is
    # the key speedup. See get_document_summary() docstring.
    try:
        doc_summary = get_document_summary(document)
    except Exception as e:
        print(f"WARNING: document summary failed for {document['source']}: {e}")
        doc_summary = document["text"][:500]  # crude fallback, still short

    results = []
    for i, chunk in enumerate(raw_chunks):
        # Per-chunk progress print so a slow/large document doesn't look
        # "frozen" in the outer per-document progress bar.
        print(f"[{document['source']}] chunk {i + 1}/{len(raw_chunks)}")
        try:
            ctx = get_chunk_context(document, doc_summary, chunk)
        except Exception as e:
            # Don't let one bad chunk kill the whole document's ingestion.
            print(f"WARNING: context generation failed for {document['source']} chunk {i}: {e}")
            ctx = ChunkContext(headline=document["source"], context="")

        page_content = "\n\n".join(
            part for part in [ctx.headline, ctx.context, chunk] if part
        )
        metadata = {
            "source": document["source"],
            "fiscal_year": document["fiscal_year"],
            "headline": ctx.headline,
            "chunk_index": i,
            "doc_chunk_count": len(raw_chunks),
        }
        results.append(Result(page_content=page_content, metadata=metadata))

    return results


def create_chunks(documents: list[dict]) -> list[Result]:
    """
    Create chunks (with contextual blurbs) using a number of workers in parallel.
    If you get a rate limit error, set WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


def create_embeddings(chunks: list[Result]):
    # Lazy import/load: only the main process (post-pool) needs the embedding
    # model, so we avoid loading a redundant copy into every worker process.
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_model.encode(texts, show_progress_bar=True).tolist()

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    # Stable IDs (source + chunk_index) instead of a plain running counter,
    # so a future incremental-ingest pass could upsert instead of rebuilding.
    ids = [f"{c.metadata['source']}::{c.metadata['chunk_index']}" for c in chunks]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")