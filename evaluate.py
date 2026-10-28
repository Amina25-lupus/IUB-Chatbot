"""
evaluate.py - Evaluate the IUB RAG chatbot (retrieval + answers).

Usage:
    python evaluate.py                    # retrieval + generation
    python evaluate.py --retrieval-only   # fast, no LLM needed
    python evaluate.py --judge            # also use an LLM as judge
    python evaluate.py --model iub-model  # compare another Ollama model
                                          # (needs get_qa_chain(model_name=...) in app.py)

Test set format (test_set.json): see test_set_template.json
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_FAISS_PATH = "faiss_index"
KS = (1, 3, 5)

REFUSAL_PHRASES = [
    "don't know", "do not know", "not in the context", "not mentioned",
    "no information", "not provided", "cannot find", "can't find",
    "couldn't find", "unable to find", "not available in", "not specified",
    "i'm sorry", "i am sorry",
]


# ----------------------------------------------------------------- helpers
def norm(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_test_set(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, item in enumerate(items, 1):
        item.setdefault("id", i)
        item.setdefault("answerable", True)
        item.setdefault("keywords", [])
        item.setdefault("source", "")
        item.setdefault("evidence", "")
        item.setdefault("answer", "")
    return items


def is_hit(doc, item):
    """A chunk is a hit if it matches the gold source file AND contains the evidence text."""
    if not item["source"] and not item["evidence"]:
        return False
    ok = True
    if item["source"]:
        ok &= norm(item["source"]) in norm(doc.metadata.get("source", ""))
    if item["evidence"]:
        ok &= norm(item["evidence"]) in norm(doc.page_content)
    return ok


def first_hit_rank(docs, item):
    for rank, doc in enumerate(docs, 1):
        if is_hit(doc, item):
            return rank
    return None


def keyword_recall(answer, keywords):
    if not keywords:
        return None
    a = norm(answer)
    return sum(norm(k) in a for k in keywords) / len(keywords)


def looks_like_refusal(answer):
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------- retrieval
def evaluate_retrieval(db, items, max_k=max(KS)):
    ranks = {}
    gradable = [it for it in items
                if it["answerable"] and (it["source"] or it["evidence"])]
    print(f"\n[Retrieval] Testing {len(gradable)} answerable questions...")
    for item in gradable:
        docs = db.similarity_search(item["question"], k=max_k)
        ranks[item["id"]] = first_hit_rank(docs, item)

    summary = {}
    for k in KS:
        summary[f"hit@{k}"] = mean(
            [1.0 if (r is not None and r <= k) else 0.0 for r in ranks.values()])
    summary["MRR"] = mean([1.0 / r if r else 0.0 for r in ranks.values()])
    return ranks, summary


# -------------------------------------------------------------- generation
def judge_answer(judge_llm, question, gold, context, answer):
    prompt = (
        "You are grading a university chatbot.\n"
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        f"Retrieved context: {context[:3000]}\n"
        f"Chatbot answer: {answer}\n\n"
        'Reply ONLY with JSON: {"correct": 0 or 1, "faithful": 0 or 1}\n'
        "correct = matches the reference answer. "
        "faithful = every claim in the answer is supported by the context."
    )
    try:
        text = judge_llm.invoke(prompt).content
        m = re.search(r"\{.*?\}", text, re.S)
        data = json.loads(m.group(0))
        return int(data.get("correct", 0)), int(data.get("faithful", 0))
    except Exception:
        return None, None


def evaluate_generation(items, ranks, model, use_judge, judge_model):
    from app import get_qa_chain

    print("\n[Generation] Loading chain...")
    chain = get_qa_chain(model) if model else get_qa_chain()

    judge_llm = None
    if use_judge:
        from langchain_ollama import ChatOllama
        judge_llm = ChatOllama(model=judge_model, temperature=0)

    rows = []
    for item in items:
        start = time.time()
        resp = chain.invoke({"input": item["question"]})
        latency = time.time() - start

        answer = resp["answer"]
        ctx_docs = resp.get("context", [])
        context_text = "\n---\n".join(d.page_content for d in ctx_docs)
        sources = sorted({Path(d.metadata.get("source", "?")).name for d in ctx_docs})

        refused = looks_like_refusal(answer)
        recall = keyword_recall(answer, item["keywords"])

        if item["answerable"]:
            auto_pass = (recall == 1.0) if recall is not None else None
        else:
            auto_pass = refused  # should say "I don't know"

        j_correct = j_faithful = None
        if judge_llm and item["answerable"]:
            j_correct, j_faithful = judge_answer(
                judge_llm, item["question"], item["answer"], context_text, answer)

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "gold_answer": item["answer"],
            "model_answer": answer.strip(),
            "retrieved_sources": "; ".join(sources),
            "retrieval_rank": ranks.get(item["id"]),
            "keyword_recall": None if recall is None else round(recall, 2),
            "refused": refused,
            "auto_pass": auto_pass,
            "latency_s": round(latency, 2),
            "judge_correct": j_correct,
            "judge_faithful": j_faithful,
            "manual_correct": "",      # fill in by hand (1/0)
            "manual_faithful": "",     # fill in by hand (1/0)
        })
        status = {True: "PASS", False: "FAIL", None: "n/a "}[auto_pass]
        print(f"  [{status}] #{item['id']} ({latency:.1f}s) {item['question'][:70]}")
    return rows


def summarize_generation(rows):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    s = {
        "keyword_recall (answerable)": mean([r["keyword_recall"] for r in ans]),
        "full_keyword_match_rate": mean(
            [1.0 if r["auto_pass"] else 0.0 for r in ans if r["auto_pass"] is not None]),
        "false_refusal_rate (answerable)": mean([1.0 if r["refused"] else 0.0 for r in ans]),
        "correct_refusal_rate (unanswerable)": mean(
            [1.0 if r["refused"] else 0.0 for r in unans]),
        "avg_latency_s": mean([r["latency_s"] for r in rows]),
    }
    jc = [r["judge_correct"] for r in ans if r["judge_correct"] is not None]
    jf = [r["judge_faithful"] for r in ans if r["judge_faithful"] is not None]
    if jc:
        s["judge_correct"] = mean(jc)
        s["judge_faithful"] = mean(jf)
    return s


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="test_set.json")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--model", default=None, help="Ollama model name override")
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge via Ollama")
    ap.add_argument("--judge-model", default="gemma2:2b")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()

    items = load_test_set(args.test_set)
    print(f"Loaded {len(items)} questions "
          f"({sum(i['answerable'] for i in items)} answerable, "
          f"{sum(not i['answerable'] for i in items)} unanswerable)")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    ranks, ret_summary = evaluate_retrieval(db, items)
    summary = {"retrieval": ret_summary}

    rows = []
    if not args.retrieval_only:
        rows = evaluate_generation(items, ranks, args.model, args.judge, args.judge_model)
        summary["generation"] = summarize_generation(rows)

    # ---- print report
    print("\n" + "=" * 52)
    print("RESULTS")
    print("=" * 52)
    for section, metrics in summary.items():
        print(f"\n{section.upper()}")
        for name, value in metrics.items():
            shown = f"{value:.2f}" if name in ("MRR", "avg_latency_s") and value is not None else pct(value)
            print(f"  {name:<38}{shown}")

    # ---- save
    Path(f"{args.out}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with open(f"{args.out}_details.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.out}_summary.json and {args.out}_details.csv")
        print("Open the CSV and fill manual_correct / manual_faithful for the real scores.")


if __name__ == "__main__":
    main()
