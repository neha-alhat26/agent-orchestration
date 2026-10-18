import asyncio
import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
HISTORY_DIR = BASE_DIR / "analysis_history"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from your .env file.")

client = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTIONS = """
You are FeedbackIQ, a customer feedback intelligence analyst.
Return only valid JSON with exactly these string fields:
summary, classification, recommendation.
classification must be exactly one of Positive, Negative, Feature Request, Other.
Positive means praise or satisfaction. Negative means a complaint or problem.
Feature Request means a request for a new capability. Other means unclear or neutral.
Do not invent facts. Keep summary and recommendation concise. No Markdown.
"""


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemini did not return a JSON object.")
    return json.loads(text[start:end + 1])


def analyze_feedback(feedback, max_retries=2):
    """One Gemini request per feedback item, with retry for temporary errors."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=f"Analyze this customer feedback:\n{feedback}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            obj = _parse_json(response.text)
            category = str(obj.get("classification", "Other")).strip().lower()
            if category.startswith("positive"):
                category = "Positive"
            elif category.startswith("negative"):
                category = "Negative"
            elif category.startswith("feature"):
                category = "Feature Request"
            else:
                category = "Other"

            summary = str(obj.get("summary", "")).strip()
            recommendation = str(obj.get("recommendation", "")).strip()
            if not summary or not recommendation:
                raise ValueError("Gemini response is missing required fields.")

            return {
                "feedback": str(feedback).strip(),
                "summary": summary,
                "classification": category,
                "recommendation": recommendation,
            }
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            transient = any(x in msg for x in (
                "503", "unavailable", "429", "resource_exhausted",
                "rate limit", "timeout", "temporarily", "500", "502", "504"
            ))
            if not transient or attempt == max_retries:
                break
            time.sleep(1.0 * (2 ** attempt) + random.uniform(0, 0.5))
    raise RuntimeError(f"Gemini analysis failed: {last_error}") from last_error


def analyze_feedback_batch(feedback_items, max_workers=3, progress_callback=None):
    """Run a small number of concurrent requests to reduce overall wait time."""
    items = [str(x).strip() for x in feedback_items if str(x).strip()]
    results, failures = [], []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 4))) as pool:
        future_map = {
            pool.submit(analyze_feedback, item): (i, item)
            for i, item in enumerate(items, start=1)
        }
        for future in as_completed(future_map):
            index, item = future_map[future]
            completed += 1
            try:
                result = future.result()
                result["id"] = index
                results.append(result)
            except Exception as exc:
                failures.append({"id": index, "feedback": item, "error": str(exc)})
            if progress_callback:
                progress_callback(completed, len(items))
    results.sort(key=lambda x: x["id"])
    failures.sort(key=lambda x: x["id"])
    return results, failures


def save_analysis(results, failures=None, source_name="Feedback"):
    """Save a run under analysis_history and update root-level latest files."""
    import io

    failures = failures or []
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem)[:35] or "analysis"
    run_id = f"{stamp}_{safe}"
    run_dir = HISTORY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    counts = {}
    for item in results:
        category = item.get("classification", "Other")
        counts[category] = counts.get(category, 0) + 1

    output = {
        "metadata": {
            "run_id": run_id,
            "analysis_name": Path(source_name).stem or "Feedback",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": MODEL,
            "total_submitted": len(results) + len(failures),
            "total_processed": len(results),
            "total_failed": len(failures),
        },
        "analytics": {
            "total_feedback": len(results),
            "positive": counts.get("Positive", 0),
            "negative": counts.get("Negative", 0),
            "feature_requests": counts.get("Feature Request", 0),
            "other": counts.get("Other", 0),
        },
        "results": results,
        "failures": failures,
    }

    def csv_text(rows):
        buf = io.StringIO()
        fields = ["id", "feedback", "summary", "classification", "recommendation"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    json_text = json.dumps(output, indent=2, ensure_ascii=False)
    (run_dir / "analysis_results.json").write_text(json_text, encoding="utf-8")
    (run_dir / "analysis_results.csv").write_text(csv_text(results), encoding="utf-8-sig")
    (BASE_DIR / "analysis_results.json").write_text(json_text, encoding="utf-8")
    (BASE_DIR / "analysis_results.csv").write_text(csv_text(results), encoding="utf-8-sig")
    return run_id, output


async def main():
    input_path = BASE_DIR / "feedback.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Create feedback.json or upload feedback in dashboard.py."
        )
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("feedback", [])
    items = []
    for item in raw:
        value = item.get("feedback") or item.get("text") or item.get("comment") or item.get("message") or "" if isinstance(item, dict) else str(item)
        if str(value).strip():
            items.append(str(value).strip())

    print(f"FeedbackIQ | Model: {MODEL} | Items: {len(items)}")
    results, failures = analyze_feedback_batch(items, max_workers=3)
    run_id, output = save_analysis(results, failures, input_path.stem)
    for item in results:
        print(f"\n{item['feedback']}\nSummary: {item['summary']}\nClassification: {item['classification']}\nAction: {item['recommendation']}")
    print("\nAnalytics:", json.dumps(output["analytics"], indent=2))
    print(f"Saved run: analysis_history/{run_id}")
    if failures:
        print(f"{len(failures)} item(s) failed:")
        for item in failures:
            print(item["id"], item["error"])


if __name__ == "__main__":
    asyncio.run(main())
