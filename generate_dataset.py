"""
IUB Q&A Dataset Generator — generate_dataset.py
Takes scraped data and generates 500+ question-answer pairs using Groq API
Run: python generate_dataset.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCRAPED_DIR     = Path("scraped_data")
OUTPUT_DIR      = Path("dataset")
TARGET_PAIRS    = 500       # minimum Q&A pairs to generate
QA_PER_CHUNK    = 8         # Q&A pairs per content chunk
MAX_CHUNK_CHARS = 3000      # max characters per chunk sent to AI
RETRY_ATTEMPTS  = 3         # retries on API failure
# ──────────────────────────────────────────────────────────────────────────────

QA_PROMPT = """You are an expert at creating AI training datasets for university chatbots.

Below is content from the official IUB (Islamia University of Bahawalpur) website.

Generate exactly {count} realistic question-answer pairs that students, applicants, or faculty might ask.

STRICT RULES:
1. Questions must sound like real students — natural and conversational
2. Answers must be complete, detailed, and based ONLY on the content provided
3. Include a mix of: simple questions, detailed procedure questions, policy questions
4. Never make up information not in the content
5. Return ONLY valid JSON — no extra text, no markdown, no explanation

REQUIRED FORMAT — return exactly this:
[
  {{
    "question": "write the question here",
    "answer": "write the complete answer here"
  }}
]

Content from IUB:
{content}"""

def call_groq_with_retry(content: str, count: int) -> list:
    """Call Groq API with retry logic."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": QA_PROMPT.format(
                        content=content[:MAX_CHUNK_CHARS],
                        count=count
                    )
                }],
                temperature=0.7,
                max_tokens=3000
            )

            text = response.choices[0].message.content.strip()

            # Extract JSON array from response
            start = text.find('[')
            end = text.rfind(']') + 1

            if start == -1 or end == 0:
                print(f"  No JSON found in response (attempt {attempt+1})")
                continue

            json_str = text[start:end]
            pairs = json.loads(json_str)

            # Validate structure
            valid_pairs = []
            for pair in pairs:
                if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                    q = str(pair['question']).strip()
                    a = str(pair['answer']).strip()
                    if len(q) > 10 and len(a) > 20:
                        valid_pairs.append({"question": q, "answer": a})

            return valid_pairs

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")

        time.sleep(2 * (attempt + 1))

    return []

def chunk_content(content: str) -> list:
    """Split long content into manageable chunks."""
    if len(content) <= MAX_CHUNK_CHARS:
        return [content]

    chunks = []
    paragraphs = content.split('\n')
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS:
            if current:
                chunks.append('\n'.join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append('\n'.join(current))

    return chunks

def load_latest_scraped_data() -> list:
    """Load the most recent scraped data file."""
    files = list(SCRAPED_DIR.glob("iub_all_data_*.json"))
    if not files:
        print("ERROR: No scraped data found. Run scraper.py first.")
        return []
    latest = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading scraped data: {latest}")
    with open(latest, 'r', encoding='utf-8') as f:
        return json.load(f)

def remove_duplicate_pairs(pairs: list) -> list:
    """Remove duplicate questions."""
    seen_questions = set()
    unique = []
    for pair in pairs:
        q_lower = pair['question'].lower().strip()
        if q_lower not in seen_questions:
            seen_questions.add(q_lower)
            unique.append(pair)
    return unique

def generate_dataset():
    print("=" * 60)
    print("IUB Q&A Dataset Generator")
    print(f"Target: {TARGET_PAIRS}+ question-answer pairs")
    print("=" * 60)

    pages = load_latest_scraped_data()
    if not pages:
        return

    print(f"Loaded {len(pages)} scraped items\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    all_pairs = []

    for i, page in enumerate(pages):
        title = page.get('title', 'Unknown')[:50]
        content = page.get('content', '')
        page_type = page.get('type', 'webpage')

        if len(content) < 200:
            print(f"[{i+1}/{len(pages)}] SKIP (too short): {title}")
            continue

        print(f"[{i+1}/{len(pages)}] {page_type.upper()}: {title}")

        chunks = chunk_content(content)
        page_pairs = []

        for j, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Chunk {j+1}/{len(chunks)}...")

            pairs = call_groq_with_retry(chunk, QA_PER_CHUNK)
            page_pairs.extend(pairs)

            # Respect Groq rate limits
            time.sleep(1.5)

        print(f"  Generated {len(page_pairs)} pairs | Total: {len(all_pairs) + len(page_pairs)}")
        all_pairs.extend(page_pairs)

        # Stop if we have enough
        if len(all_pairs) >= TARGET_PAIRS * 2:
            print(f"\nReached {len(all_pairs)} pairs — stopping early")
            break

    # Remove duplicates
    print(f"\nRemoving duplicates from {len(all_pairs)} pairs...")
    all_pairs = remove_duplicate_pairs(all_pairs)
    print(f"After deduplication: {len(all_pairs)} pairs")

    if len(all_pairs) == 0:
        print("ERROR: No pairs generated. Check your API key.")
        return

    # Save as JSON (readable)
    json_file = OUTPUT_DIR / "iub_qa_dataset.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    # Save as JSONL (for fine-tuning)
    jsonl_file = OUTPUT_DIR / "iub_qa_dataset.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            # Format for instruction fine-tuning
            record = {
                "text": (
                    f"<|user|>\n{pair['question']}\n"
                    f"<|assistant|>\n{pair['answer']}"
                )
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "total_pairs": len(all_pairs),
        "source_pages": len(pages),
        "json_file": str(json_file),
        "jsonl_file": str(jsonl_file)
    }
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print(f"  Total Q&A pairs : {len(all_pairs)}")
    print(f"  JSON file       : {json_file}")
    print(f"  JSONL file      : {jsonl_file}")
    print(f"\nNEXT STEP: Upload {jsonl_file} to Google Colab for fine-tuning")
    print("=" * 60)

if __name__ == "__main__":
    generate_dataset()
