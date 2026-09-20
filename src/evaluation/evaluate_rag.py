import sys
import json
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.rag.rag_pipeline import rag_pipeline
QUESTIONS_FILE = PROJECT_ROOT / "src" / "evaluation" / "evaluation_questions.json"
RESULTS_FILE = PROJECT_ROOT / "src" / "evaluation" / "evaluation_results.json"
def load_questions():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)
def evaluate():
    questions = load_questions()
    results = []
    total_start = time.time()
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        category = item["category"]
        print(f"\n[{index}/{len(questions)}] {question}")
        start_time = time.time()
        try:
            result = rag_pipeline(
                question,
                top_k=5
            )
            processing_time = round(
                time.time() - start_time,
                3
            )
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            fallback = (
                answer.strip()
                == "I don't know based on the provided documents."
            )
            results.append({
                "question": question,
                "category": category,
                "answer": answer,
                "source_count": len(sources),
                "fallback": fallback,
                "processing_time_seconds": processing_time
            })
            print(f"Answer: {answer[:200]}")
            print(
                f"Sources: {len(sources)} | "
                f"Time: {processing_time}s"
            )
        except Exception as error:
            processing_time = round(
                time.time() - start_time,
                3
            )
            results.append({
                "question": question,
                "category": category,
                "answer": "",
                "source_count": 0,
                "fallback": True,
                "processing_time_seconds": processing_time,
                "error": str(error)
            })
            print(f"ERROR: {error}")
    total_time = round(
        time.time() - total_start,
        3
    )
    successful = sum(
        1
        for result in results
        if not result["fallback"]
    )
    total = len(results)
    success_rate = round(
        (successful / total) * 100,
        2
    ) if total else 0
    average_time = round(
        sum(
            result["processing_time_seconds"]
            for result in results
        ) / total,
        3
    ) if total else 0
    evaluation_summary = {
        "total_questions": total,
        "successful_answers": successful,
        "fallback_answers": total - successful,
        "success_rate_percent": success_rate,
        "average_processing_time_seconds": average_time,
        "total_evaluation_time_seconds": total_time,
        "results": results
    }
    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            evaluation_summary,
            file,
            indent=4,
            ensure_ascii=False
        )
    print("\n" + "=" * 70)
    print("RAG EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total Questions: {total}")
    print(f"Successful Answers: {successful}")
    print(f"Fallback Answers: {total - successful}")
    print(f"Success Rate: {success_rate}%")
    print(f"Average Processing Time: {average_time}s")
    print(f"Results saved to: {RESULTS_FILE}")
if __name__ == "__main__":
    evaluate()
