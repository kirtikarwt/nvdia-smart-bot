from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv(override=True)


MODEL = "ollama/llama3.2"
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "data"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

COLLECTION_NAME = "docs"


EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

wait = wait_exponential(multiplier=1, min=10, max=240)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

RETRIEVAL_K = 30
FINAL_K = 10

RERANK_CONTEXT_CHAR_BUDGET = 12000

SYSTEM_PROMPT = """
You are a knowledgeable assistant representing NVIDIA.

Answer the user's question ONLY using the retrieved context.

IMPORTANT RULES:
1. Pay close attention to the years requested by the user.
2. Never substitute one year for another.
3. If the requested year is not present in the retrieved context, say that the information was not retrieved.
4. Do not invent financial figures.
5. For financial questions, clearly distinguish revenue, net income,
   operating income, cash flow, assets, liabilities, and other financial metrics.
6. If multiple years are requested, compare those exact years.
7. If no context was retrieved at all, say plainly that nothing relevant was found in the
   knowledge base rather than attempting to answer from general knowledge.

For context, here are extracts from the NVIDIA Knowledge Base:

{context}

Answer the user's question accurately and completely.
"""


class Result(BaseModel):
    id: str
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def get_embedding_model():
    """
    Lazily load and cache the embedding model on first use. Kept as a
    function (rather than a module-level global) so importing this module
    doesn't force-load the model, and so tests/other entry points can mock
    it easily.
    """
    from sentence_transformers import SentenceTransformer

    if not hasattr(get_embedding_model, "_model"):
        get_embedding_model._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return get_embedding_model._model


def embed_query(text: str) -> list[float]:
    """Embed a *query* string. BGE models need an instruction prefix on the
    query side only — passages/chunks were embedded without one at ingestion
    time, so don't add the prefix anywhere else."""
    model = get_embedding_model()
    return model.encode(QUERY_INSTRUCTION + text).tolist()


@retry(wait=wait, stop=stop_after_attempt(5))
def rerank(question, chunks):
    if not chunks:
        return []

    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"

    # Cap total context so we don't exceed a local model's context window.
    budget = RERANK_CONTEXT_CHAR_BUDGET
    included = []
    for index, chunk in enumerate(chunks):
        text = f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
        if budget - len(text) < 0 and included:
            # Stop adding chunks once we'd exceed budget, but always include
            # at least one chunk even if it alone exceeds the budget.
            break
        user_prompt += text
        included.append(index + 1)
        budget -= len(text)

    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order

    # Validate the LLM's returned order before trusting it: it may
    # hallucinate ids outside range, drop ids, or duplicate them. If the
    # order isn't a clean permutation of what we sent, fall back to the
    # original retrieval order for the chunks we included, then append any
    # excluded (over-budget) chunks at the end.
    valid_ids = set(included)
    seen = set()
    clean_order = []
    for i in order:
        if i in valid_ids and i not in seen:
            clean_order.append(i)
            seen.add(i)

    if set(clean_order) != valid_ids:
        # Reranking response was malformed; don't lose chunks, just fall
        # back to retrieval order for anything the LLM didn't cleanly rank.
        remaining = [i for i in included if i not in seen]
        clean_order = clean_order + remaining

    reranked = [chunks[i - 1] for i in clean_order]
    # Append any chunks that were excluded from the rerank call due to the
    # character budget, preserving their original retrieval order.
    excluded = [c for idx, c in enumerate(chunks) if (idx + 1) not in included]
    return reranked + excluded


def make_rag_messages(question, history, chunks):
    if chunks:
        context = "\n\n".join(
            f"Extract from {chunk.metadata['source']} (fiscal year {chunk.metadata.get('fiscal_year', 'unknown')}):\n{chunk.page_content}"
            for chunk in chunks
        )
    else:
        context = "(No relevant context was retrieved from the knowledge base.)"
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop_after_attempt(5))
def rewrite_query(question, history=None):
    if history is None:
        history = []

    # Convert Gradio history into simple text
    history_text = ""
    for msg in history:
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"Assistant: {msg['content']}\n"

    prompt = f"""
You rewrite user questions for a company knowledge base.

The knowledge base contains information about NVIDIA, covering fiscal years
2023, 2024, and 2025, including:
- financial statements
- annual reports / 10-K filings
- proxy statements and shareholder letters
- products
- revenue
- risks
- management
- strategy
- business segments

Conversation history:
{history_text}

Current user question:
{question}

Rewrite the current question into ONE short, precise search query.

Rules:
1. Use conversation history ONLY when the current question depends on it.
2. Resolve references such as "it", "they", "this", "that", or "2023".
3. Do NOT answer the question.
4. Do NOT talk about the conversation or history.
5. Do NOT add information that isn't present.
6. If the question is already clear, return it almost unchanged.
7. Return ONLY the rewritten search query.

Examples:

User: What was NVIDIA's revenue in 2025?
Query: NVIDIA revenue 2025

User: How did it compare with 2023?
Query: NVIDIA revenue comparison 2025 2023

User: What about its data center business?
Query: NVIDIA data center business revenue 2025

User: What is the gross margin?
Query: NVIDIA gross margin

Current question:
{question}
"""

    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def merge_chunks(chunks, reranked):
    """Merge two chunk lists, deduping on the stable Chroma id rather than
    on page_content text (id is the reliable key; ingestion assigns it as
    source::chunk_index)."""
    merged = chunks[:]
    existing_ids = {chunk.id for chunk in chunks}
    for chunk in reranked:
        if chunk.id not in existing_ids:
            merged.append(chunk)
            existing_ids.add(chunk.id)
    return merged


def fetch_context_unranked(question):
    query = embed_query(question)

    results = collection.query(
        query_embeddings=[query],
        n_results=RETRIEVAL_K
    )

    chunks = []
    for result_id, doc, meta in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        chunks.append(Result(id=result_id, page_content=doc, metadata=meta))

    return chunks


def fetch_context(original_question, history=None):
    rewritten_question = rewrite_query(original_question, history)
    print(rewritten_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait, stop=stop_after_attempt(3))
def answer_question(question: str, history: list[dict] | None = None):
    """
    Stream the answer using RAG.
    """
    history = history or []

    chunks = fetch_context(question, history)

    messages = make_rag_messages(
        question,
        history,
        chunks
    )

    response = completion(
        model=MODEL,
        messages=messages,
        stream=True
    )

    answer = ""

    for chunk in response:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            yield answer, chunks