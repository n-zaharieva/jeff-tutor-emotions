"""
Baseline evaluation: Jeff's ready-made emotion adapter on tutor messages.

The emotion adapter (jeff.language.emotion) picks one of 27 emotions (or neutral),
learned from general online comments. A tutor needs different labels, so each of
the 28 emotions is mapped to one of six tutor labels below. The script then
measures how often the mapped answer matches the label in tutor_emotions.csv.

Run from inside the jeff folder:
    uv run python evaluate_baseline.py

Output:
    - a summary in the terminal
    - results_baseline.csv with one row per test message
"""

import csv
import time
from collections import Counter, defaultdict

import jeff

DATA_FILE = "tutor_emotions.csv"
RESULTS_FILE = "results_baseline.csv"
SPLIT = "test"  # evaluate on the held-out messages only

# Mapping from the adapter's 28 labels to the six tutor labels.
# There is no "bored" emotion among the 28, so the baseline can never predict
# "bored". That is a limitation of the ready-made adapter, and one reason to
# train a tutor-specific adapter.
EMOTION_TO_TUTOR = {
    # frustrated
    "anger": "frustrated", "annoyance": "frustrated", "disappointment": "frustrated",
    "disapproval": "frustrated", "disgust": "frustrated", "embarrassment": "frustrated",
    # confused
    "confusion": "confused",
    # confident / positive
    "admiration": "confident", "amusement": "confident", "approval": "confident",
    "excitement": "confident", "gratitude": "confident", "joy": "confident",
    "love": "confident", "optimism": "confident", "pride": "confident",
    "relief": "confident", "realization": "confident",
    # neutral
    "neutral": "neutral", "curiosity": "neutral", "surprise": "neutral",
    "desire": "neutral", "caring": "neutral",
    # distressed
    "fear": "distressed", "nervousness": "distressed", "sadness": "distressed",
    "grief": "distressed", "remorse": "distressed",
}


def load_messages(path, split):
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


def main():
    rows = load_messages(DATA_FILE, SPLIT)
    print(f"Loaded {len(rows)} {SPLIT} messages from {DATA_FILE}\n")

    client = jeff.Jeff.local(["emotion"])
    emotion = client.function("jeff.language.emotion.emotion", context={
        "source": "A student's message to an online tutor",
    })

    results = []
    for row in rows:
        start = time.perf_counter()
        d = emotion(row["text"])
        ms = (time.perf_counter() - start) * 1000

        predicted = EMOTION_TO_TUTOR.get(d.value, "neutral")
        results.append({
            "id": row["id"],
            "text": row["text"],
            "difficulty": row["difficulty"],
            "true_label": row["label"],
            "jeff_emotion": d.value,
            "predicted_label": predicted,
            "confidence": round(d.confidence, 3) if d.confidence is not None else "",
            "abstained": d.abstained,
            "correct": predicted == row["label"],
            "ms": round(ms, 1),
        })

    # Save every prediction so you can quote examples in your report.
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
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
