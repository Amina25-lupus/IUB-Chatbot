"""
generate_test_set.py - Builds test_set.json automatically from your own FAISS index.

Usage (Ollama must be running):
    python generate_test_set.py                    # 60 answerable questions
    python generate_test_set.py --n 80 --model llama3:8b   # bigger model = better questions

Then run:
    python evaluate.py --retrieval-only
    python evaluate.py
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from tqdm import tqdm

DB_FAISS_PATH = "faiss_index"
MAX_PER_SOURCE = 8       # keeps one big document from dominating the test set
MIN_CHUNK_CHARS = 300    # skip tiny / heading-only chunks

BAD_WORDS = re.compile(r"\b(passage|the text|this text|document|context|above|provided)\b", re.I)

PROMPT = """You are writing exam questions for a university chatbot.
Read the passage and write ONE question a student might really ask, whose answer is clearly stated in the passage.

Rules:
- The question must make sense on its own. Never say "the passage", "the text" or "the document". Name the program, department or topic so it is unambiguous.
- "answer": 1-2 sentences, using only facts from the passage.
- "keywords": 1-3 short strings copied EXACTLY from the passage (numbers, dates, names, amounts) that any correct answer must contain.
- "evidence": an exact phrase of 4-10 words copied from the passage that contains the answer.
- If the passage has no concrete fact (only headings, boilerplate, contact lists), return {{"skip": true}}.

Return only JSON: {{"question": "...", "answer": "...", "keywords": ["..."], "evidence": "..."}}

Passage:
{chunk}
"""

# Questions the documents should NOT be able to answer. The bot should say "I don't know".
UNANSWERABLE = [
    "Who is the captain of the IUB cricket team?",
    "What is the Wi-Fi password of the IUB library?",
    "What is the fee structure of Punjab University Lahore?",
    "Can you recommend a good pizza place near the university?",
    "Who won the last FIFA World Cup?",
    "What is the weather in Bahawalpur tomorrow?",
    "How do I install Python on Windows?",
    "What is the capital of France?",
    "What is the phone number of the vice chancellor's personal secretary at Harvard University?",
    "Which bus company runs the IUB Bahawalpur to Karachi route and what is the ticket price?",
    "IUB mein cricket ground kitne acre ka hai aur uska booking charge kya hai?",
    "Mujhe aaj raat ke khane ke liye biryani ki recipe batao.",
]


def norm(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def pick_chunks(db, n, seed):
    docs = [d for d in db.docstore._dict.values() if len(d.page_content) >= MIN_CHUNK_CHARS]
    random.Random(seed).shuffle(docs)
    per_source = defaultdict(int)
    picked = []
    for d in docs:
        src = d.metadata.get("source", "?")
        if per_source[src] < MAX_PER_SOURCE:
            per_source[src] += 1
            picked.append(d)
    # return more than n so we can afford rejections
    return picked[: n * 3]


def valid(item, chunk_text):
    try:
        q = item["question"].strip()
        if BAD_WORDS.search(q) or len(q) < 15:
            return False
        if not item["answer"].strip():
            return False
        ct = norm(chunk_text)
        if not item["keywords"] or not all(norm(k) in ct for k in item["keywords"]):
            return False
        if norm(item["evidence"]) not in ct:
            return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def generate_one(llm, chunk_text, tries=2):
    for _ in range(tries):
        try:
            raw = llm.invoke(PROMPT.format(chunk=chunk_text[:1500])).content
            item = json.loads(raw)
            if item.get("skip"):
                return None
            if valid(item, chunk_text):
                return item
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of answerable questions")
    ap.add_argument("--model", default="gemma2:2b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="test_set.json")
    args = ap.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOllama(model=args.model, temperature=0.2, format="json")

    candidates = pick_chunks(db, args.n, args.seed)
    print(f"{len(candidates)} candidate chunks, generating up to {args.n} questions with {args.model}...")

    items, seen_q = [], set()
    for chunk in tqdm(candidates):
        if len(items) >= args.n:
            break
        item = generate_one(llm, chunk.page_content)
        if not item or norm(item["question"]) in seen_q:
            continue
        seen_q.add(norm(item["question"]))
        items.append({
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
            "source": Path(chunk.metadata.get("source", "")).name,
            "evidence": item["evidence"].strip(),
            "keywords": [k.strip() for k in item["keywords"]],
            "answerable": True,
        })

    for q in UNANSWERABLE:
        items.append({
            "question": q, "answer": "Not in the documents. The bot should say it doesn't know.",
            "source": "", "evidence": "", "keywords": [], "answerable": False,
        })

    for i, it in enumerate(items, 1):
        it["id"] = i

    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ans = sum(i["answerable"] for i in items)
    print(f"\nSaved {args.out}: {n_ans} answerable + {len(items) - n_ans} unanswerable questions.")
    if n_ans < args.n:
        print("Fewer questions than requested passed validation. Try --model llama3:8b or a higher --n.")


if __name__ == "__main__":
    main()
