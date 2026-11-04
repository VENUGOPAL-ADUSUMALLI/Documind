"""Retrieval evaluation: does the search put the right page in the top results?

How it works:
  1. Build a temporary index (in memory) from your PDFs, using the real upload pipeline.
  2. For every question in questions.json, run the real search.
  3. Check if the expected (document, page) is among the top results.
  4. Print hit-rate and MRR for each setting, so settings can be compared.
  5. Then ask the LLM the questions that list "facts", and check how many of those
     facts appear in the answer (completeness), for different context-window sizes.
     (dense = meaning search, bm25 = keyword search, hybrid = both merged.)

Run it from the backend folder:
    python -m eval.run_eval --docs /path/to/folder/with/the/pdfs
"""

import argparse
import json
from pathlib import Path

from app import rag
from app.config import settings
from app.db import Base, make_engine, make_session_factory
from app.embeddings import get_embedder
from app.ingestion import ingest_pdf
from app.llm import OpenAIChat
from app.reranker import CrossEncoderReranker
from app.vector_store import make_qdrant_client

TOP_K = 5  # how many results count ("hit-rate@5")

# The settings we compare. Add a line here to test another idea.
CONFIGS = [
    {"name": "dense only",         "mode": "dense",  "rerank": False},
    {"name": "keywords (BM25) only", "mode": "bm25", "rerank": False},
    {"name": "hybrid (RRF)",       "mode": "hybrid", "rerank": False},
    {"name": "dense + reranker",   "mode": "dense",  "rerank": True},
    {"name": "hybrid + reranker",  "mode": "hybrid", "rerank": True},
]

# All settings above share one index, built with these chunk settings.
# Context-window sizes compared in the answer-completeness test.
WINDOWS = [("matched chunk only", 0), ("window +-1 chunk", 1), ("window +-2 chunks", 2)]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def build_index(docs_dir, filenames, embedder, chunk_size, chunk_overlap):
    """Upload all the PDFs into a throwaway database + Qdrant. Returns (session, qdrant)."""
    # The chunking code reads these two numbers from settings, so set them first.
    settings.chunk_size = chunk_size
    settings.chunk_overlap = chunk_overlap

    engine = make_engine("sqlite://")  # a database that lives only in memory
    Base.metadata.create_all(engine)
    session = make_session_factory(engine)()
    qdrant = make_qdrant_client(":memory:")

    for filename in filenames:
        pdf_bytes = (docs_dir / filename).read_bytes()
        ingest_pdf(session, qdrant, embedder, filename, pdf_bytes)
    return session, qdrant


def find_rank(sources, question):
    """Position (1 = best) of the expected page in the results, or None if missing."""
    for position, source in enumerate(sources, start=1):
        if source["filename"] == question["document"] and source["page"] in question["pages"]:
            return position
    return None


def evaluate(session, qdrant, embedder, reranker, questions):
    """Run every answerable question. Returns one result dict per question."""
    results = []
    for question in questions:
        sources = rag.retrieve(session, qdrant, embedder, question["question"], None, reranker)
        results.append(
            {
                "question": question["question"],
                "rank": find_rank(sources, question),
                "top": f'{sources[0]["filename"]} p{sources[0]["page"]}' if sources else "nothing",
            }
        )
    return results


def count_facts(answer, facts):
    """How many required facts appear in the answer?

    A fact like "T-of-N | threshold" counts if ANY of its alternatives is in the answer.
    """
    text = answer.lower()
    found = 0
    for fact in facts:
        if any(option.strip().lower() in text for option in fact.split("|")):
            found += 1
    return found


def evaluate_answers(session, qdrant, embedder, llm, reranker, questions):
    """Ask the LLM each question. Returns {question: (facts_found, facts_total, context_chars)}."""
    outcome = {}
    for question in questions:
        result = rag.answer_question(session, qdrant, embedder, llm, question["question"], None, reranker)
        context_chars = sum(len(source["excerpt"]) for source in result["sources"])
        found = count_facts(result["answer"], question["facts"])
        outcome[question["question"]] = (found, len(question["facts"]), context_chars)
    return outcome


def summarize(results):
    """hit-rate = share of questions with the right page in the top results.
    MRR = average of 1/rank (a miss counts as 0)."""
    hits = sum(1 for r in results if r["rank"] is not None)
    mrr = sum(1 / r["rank"] for r in results if r["rank"] is not None) / len(results)
    return hits, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="eval/docs", help="folder containing the PDFs")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--skip-answers", action="store_true", help="only run the retrieval part")
    args = parser.parse_args()

    docs_dir = Path(args.docs).expanduser()
    questions = json.loads(Path(args.questions).read_text())
    answerable = [q for q in questions if q["document"]]
    unanswerable = [q for q in questions if not q["document"]]
    filenames = sorted({q["document"] for q in answerable})

    settings.retrieval_top_k = TOP_K
    embedder = get_embedder()
    reranker = CrossEncoderReranker()
    llm = OpenAIChat()

    session, qdrant = build_index(docs_dir, filenames, embedder, CHUNK_SIZE, CHUNK_OVERLAP)
    summary_rows = []
    all_results = {}

    for config in CONFIGS:
        settings.retrieval_mode = config["mode"]  # which search to use for this run
        results = evaluate(session, qdrant, embedder, reranker if config["rerank"] else None, answerable)
        hits, mrr = summarize(results)
        summary_rows.append((config["name"], hits, len(results), mrr))
        all_results[config["name"]] = results

    print(f"\nRetrieval results ({len(answerable)} questions, top {TOP_K})\n")
    print(f'{"setting":24} {"hit-rate@5":>12} {"MRR":>6}')
    for name, hits, total, mrr in summary_rows:
        print(f"{name:24} {f'{hits}/{total} ({hits / total:.0%})':>12} {mrr:6.2f}")

    print("\nPer question (rank of the correct page; '-' = not found in top 5):")
    names = [config["name"] for config in CONFIGS]
    for index, question in enumerate(answerable):
        ranks = [all_results[name][index]["rank"] for name in names]
        shown = "  ".join(f'{(r or "-")!s:>2}' for r in ranks)
        print(f'  {shown}   {question["question"]}')
    print("  (columns, in order: " + " | ".join(names) + ")")

    # Questions with no answer in the documents: show what the app says. Judge by eye.
    settings.retrieval_mode = "hybrid"
    settings.context_window = 0
    for question in unanswerable:
        answer = rag.answer_question(session, qdrant, embedder, llm, question["question"], None, reranker)
        print(f'\nNot in documents: "{question["question"]}"\n  app says: {answer["answer"]}')

    if args.skip_answers:
        return

    # Completeness: does the answer contain all the facts it should?
    with_facts = [q for q in answerable if q.get("facts")]
    settings.retrieval_mode = "hybrid"
    outcomes = {}
    for name, window in WINDOWS:
        settings.context_window = window
        outcomes[name] = evaluate_answers(session, qdrant, embedder, llm, reranker, with_facts)

    print(f"\nAnswer completeness ({len(with_facts)} questions, hybrid + reranker, top {TOP_K})\n")
    print(f'{"context sent to the LLM":26} {"facts found":>12} {"avg context chars":>18}')
    for name, _ in WINDOWS:
        rows = outcomes[name].values()
        found = sum(r[0] for r in rows)
        total = sum(r[1] for r in rows)
        average_chars = sum(r[2] for r in rows) / len(rows)
        print(f"{name:26} {f'{found}/{total} ({found / total:.0%})':>12} {average_chars:18.0f}")

    print("\nPer question (facts found / facts required):")
    for question in with_facts:
        cells = []
        for name, _ in WINDOWS:
            found, total, _chars = outcomes[name][question["question"]]
            cells.append(f"{found}/{total}")
        print(f'  {"   ".join(f"{c:>4}" for c in cells)}   {question["question"]}')
    print("  (columns, in order: " + " | ".join(name for name, _ in WINDOWS) + ")")


if __name__ == "__main__":
    main()
