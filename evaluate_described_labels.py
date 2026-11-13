"""
Experiment 2: Jeff with the tutor's own labels, described in words (no training).

Jeff's triage adapter routes a message to one of the caller's own "teams", reading
a one-line description of each. Here the "teams" are the six tutor responses, and
each description says which student state it is for. The model is not trained on
tutor data; only the wording of the labels changes. This tests whether describing
the labels well is enough, compared with the ready-made emotion adapter
(evaluate_baseline.py).

Run from inside the jeff folder:
    uv run python evaluate_described_labels.py

Output:
    - a summary in the terminal
    - results_described_labels.csv with one row per test message
"""

import csv
import time
from collections import Counter, defaultdict

from jeff import Jeff

DATA_FILE = "tutor_emotions.csv"
RESULTS_FILE = "results_described_labels.csv"
SPLIT = "test"  # the same 60 held-out messages as the baseline

# The label keys match the labels in tutor_emotions.csv, so no mapping is needed.
# Each description names both the student's state and what the tutor should do,
# because the triage adapter was trained to pick who should handle a message.
TUTOR_LABELS = {
    "frustrated": "Encouragement: the student is frustrated, annoyed or angry about getting things wrong, "
                  "or criticises themselves",
    "confused": "Explanation: the student does not understand something and needs it explained again",
    "bored": "Engagement: the student is bored, or finds the work repetitive, pointless or too easy",
    "confident": "Next challenge: the student understood, got it right or feels confident and is ready to move on",
    "neutral": "Regular tutoring: a plain question, answer or request with no strong emotion",
    "distressed": "Human counsellor: the student is anxious, panicking, overwhelmed, hopeless or unwell, "
                  "beyond normal frustration with an exercise",
}

CONTEXT = {
    "company": "An online tutoring service where university students learn calculus and Dutch with an AI tutor.",
    "channel": "chat",
}


def load_messages(path, split):
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


def main():
    rows = load_messages(DATA_FILE, SPLIT)
    print(f"Loaded {len(rows)} {SPLIT} messages from {DATA_FILE}\n")

    jeff = Jeff.local(["triage"])
    route = jeff.function("triage.route", context=CONTEXT, labels=TUTOR_LABELS)

    results = []
    for row in rows:
        start = time.perf_counter()
        d = route(row["text"])
        ms = (time.perf_counter() - start) * 1000

        results.append({
            "id": row["id"],
            "text": row["text"],
            "difficulty": row["difficulty"],
            "true_label": row["label"],
            "predicted_label": d.value,
            "confidence": round(d.confidence, 3) if d.confidence is not None else "",
            "abstained": d.abstained,
            "correct": d.value == row["label"],
            "ms": round(ms, 1),
        })

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"Overall accuracy: {correct}/{n} = {correct / n:.0%}")

    confident = [r for r in results if not r["abstained"]]
    unsure = [r for r in results if r["abstained"]]
    if confident:
        acc = sum(r["correct"] for r in confident) / len(confident)
        print(f"When Jeff was sure ({len(confident)} messages): {acc:.0%} correct")
    if unsure:
        acc = sum(r["correct"] for r in unsure) / len(unsure)
        print(f"When Jeff was unsure ({len(unsure)} messages): {acc:.0%} correct")

    times = sorted(r["ms"] for r in results)
    print(f"Median time per message: {times[len(times) // 2]:.0f} ms\n")

    print("Accuracy per tutor label:")
    by_label = defaultdict(list)
    for r in results:
        by_label[r["true_label"]].append(r["correct"])
    for label, values in by_label.items():
        print(f"  {label:<11} {sum(values)}/{len(values)}")

    print("\nAccuracy per message type:")
    by_type = defaultdict(list)
    for r in results:
        by_type[r["difficulty"]].append(r["correct"])
    for kind, values in by_type.items():
        print(f"  {kind:<11} {sum(values)}/{len(values)}")

    print("\nMost common mistakes (true label -> predicted label):")
    mistakes = Counter((r["true_label"], r["predicted_label"]) for r in results if not r["correct"])
    for (true, pred), count in mistakes.most_common(5):
        print(f"  {true} -> {pred}: {count}")

    print(f"\nAll predictions saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
