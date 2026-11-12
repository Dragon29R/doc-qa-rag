"""Automatic evaluation for the RAG pipeline.

The pipeline is passed in as a function, so this file does not depend on where the
pipeline lives (notebook now, rag.py later):

    pipeline(question: str) -> {"answer": str, "chunks": [{"id", "source", "page", ...}, ...]}

Scores
  - retrieval_hit (answerable questions): for EVERY expected group, at least one retrieved
    chunk comes from one of the group's (source, PDF page) pairs. A two-part question has
    two groups, so a miss on one half is visible.
  - refused (negative questions): the answer contains a refusal phrase.
Informational: cited_hit (are the expected pages cited in the answer text), n_chunks sent
to the model. Retrieval hit does NOT mean the answer is correct: read the answers too.
"""
import contextlib
import csv
import io
import json
import re
import time
from datetime import datetime
from pathlib import Path

REFUSAL_MARKERS = ("can't answer", "cannot answer", "can not answer",
                   "i don't know", "não sei", "nao sei")
CITATION_RE = re.compile(r"\[File:\s*([^,\]]+?)\s*,\s*Page:\s*(\d+)\s*\]")


def load_eval_set(path="eval_set.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_refusal(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return any(marker in a for marker in REFUSAL_MARKERS)


def cited_pairs(answer: str) -> set:
    return {(s.strip(), int(p)) for s, p in CITATION_RE.findall(answer)}


def groups_hit(groups, pairs) -> list:
    """One bool per expected group: was at least one of its pages in `pairs`?"""
    return [any((g["source"], p) in pairs for p in g["pages"]) for g in groups]


RETRY_RE = re.compile(r"retry in ([\d.]+)s", re.I)


def call_with_retry(pipeline, question, max_retries=3, default_wait=45):
    """Call the pipeline silently; on a 429 / RESOURCE_EXHAUSTED wait the delay the API asks for."""
    for attempt in range(max_retries + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # silence the pipeline's prints
                return pipeline(question)
        except Exception as e:
            msg = str(e)
            limited = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if not limited or attempt == max_retries:
                raise
            m = RETRY_RE.search(msg)
            wait = (float(m.group(1)) if m else default_wait) + 3
            print(f"  rate limited, waiting {wait:.0f}s (retry {attempt + 1}/{max_retries})")
            time.sleep(wait)


def run_eval(pipeline, label, eval_path="eval_set.json", out_dir="results", pause=8.0):
    items = load_eval_set(eval_path)
    todo = [i for i in items if i["answerable"] and not i["expected_groups"]]
    if todo:
        raise ValueError(f"expected_groups missing for questions {[i['id'] for i in todo]}")

    rows = []
    for item in items:
        t0 = time.time()
        try:
            out = call_with_retry(pipeline, item["question"])
            answer, chunks, error = out["answer"], out["chunks"], None
        except Exception as e:                                 # keep going, record the failure
            answer, chunks, error = "", [], f"{type(e).__name__}: {e}"

        retrieved = {(c["source"], c["page"]) for c in chunks}
        cited = cited_pairs(answer)
        row = {
            "id": item["id"], "question": item["question"], "answerable": item["answerable"],
            "answer": answer, "error": error, "seconds": round(time.time() - t0, 1),
            "n_chunks": len(chunks),
            "retrieved": [{"id": c["id"], "page": c["page"], "source": c["source"]} for c in chunks],
            "cited": sorted(cited),
        }
        if item["answerable"]:
            r = groups_hit(item["expected_groups"], retrieved)
            c = groups_hit(item["expected_groups"], cited)
            row.update(groups_retrieved=r, retrieval_hit=all(r), cited_hit=all(c),
                       passed=all(r) and error is None)
        else:
            row.update(refused=is_refusal(answer), passed=is_refusal(answer) and error is None)
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'}  Q{item['id']:<2} "
              f"chunks={row['n_chunks']:<3} {row['seconds']}s  {error or ''}")
        time.sleep(pause)

    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    summary = {
        "label": label, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "retrieval_hit": f"{sum(r['retrieval_hit'] for r in ans)}/{len(ans)}",
        "cited_hit": f"{sum(r['cited_hit'] for r in ans)}/{len(ans)}",
        "negatives_refused": f"{sum(r['refused'] for r in neg)}/{len(neg)}",
        "avg_chunks": round(sum(r["n_chunks"] for r in rows) / len(rows), 1),
        "errors": sum(r["error"] is not None for r in rows),
    }

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    (out_path / f"{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_path / "summary.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)

    print("\n| run | retrieval hit | cited hit | negatives refused | avg chunks | errors |")
    print("|---|---|---|---|---|---|")
    print(f"| {label} | {summary['retrieval_hit']} | {summary['cited_hit']} | "
          f"{summary['negatives_refused']} | {summary['avg_chunks']} | {summary['errors']} |")
    return summary, rows