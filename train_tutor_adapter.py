"""
Experiment 3: fine-tune Jeff's base model on tutor messages (LoRA).

Jeff's own trainer (jeff-train) is not included in the public repository, so this
script fine-tunes Jeff's base model with standard LoRA (Hugging Face PEFT), copying
Jeff's core idea: the model reads a prompt that lists the options as letters, and
the answer is read from the model's probabilities for those letters in a single
forward pass (no generated text).

Steps:
  1. Load Jeff's base model and the messages in tutor_emotions.csv.
  2. Evaluate the untrained base on the 60 test messages (before training).
  3. Train a small LoRA adapter on the 120 training messages.
  4. Evaluate again on the same 60 test messages (after training).
  5. Save the adapter to tutor_adapter/ and every prediction to results_finetuned.csv.

Run from inside the jeff folder (Apple silicon uses the GPU through "mps"): uv run --with peft --with transformers --with accelerate python train_tutor_adapter.py
"""

import csv
import random
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer


BASE_MODEL = "mstrasser/jeff-base"   # Jeff's base model on Hugging Face
BASE_REVISION = "v1.4"               # the base version your Jeff install uses

DATA_FILE = "tutor_emotions.csv"
RESULTS_FILE = "results_finetuned.csv"
ADAPTER_DIR = "tutor_adapter"

EPOCHS = 3
ACCUMULATE = 4          # messages per weight update (one message at a time saves memory)
LEARNING_RATE = 2e-4
LORA_RANK = 16          # the same rank Jeff's own adapters use
SEED = 42

# LoRA is added only to the top 8 of the text model's 24 layers (16-23), not to the image part.
LORA_TARGETS = (r".*language_model\.layers\.(1[6-9]|2[0-3])\..*\.(q_proj|k_proj|v_proj|o_proj"
                r"|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)")

# The six tutor labels, with the same descriptions as experiment 2.
LABELS = {
    "frustrated": "the student is frustrated, annoyed or angry about getting things wrong, or criticises themselves",
    "confused": "the student does not understand something and needs it explained again",
    "bored": "the student is bored, or finds the work repetitive, pointless or too easy",
    "confident": "the student understood, got it right or feels confident and is ready to move on",
    "neutral": "a plain question, answer or request with no strong emotion",
    "distressed": "the student is anxious, panicking, overwhelmed, hopeless or unwell, beyond normal frustration",
}
LABEL_NAMES = list(LABELS)
LETTERS = ["A", "B", "C", "D", "E", "F"]

# Setup
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_rows(split):
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["split"] == split]


def build_prompt(message):
    """The prompt lists the options as letters, like Jeff's own requests."""
    options = "\n".join(f"{LETTERS[i]}. {name}: {LABELS[name]}" for i, name in enumerate(LABEL_NAMES))
    return (
        "You read the messages students send to an AI tutor for university calculus and Dutch.\n"
        "Decide which option best describes the student's state.\n\n"
        f"Options:\n{options}\n\n"
        f"Student message: {message}\n"
        "Answer:"
    )


def load_model():
    """Load Jeff's base and stop if its weights did not load (a silent failure gives random answers)."""
    print(f"Loading {BASE_MODEL} ({BASE_REVISION}) ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model, info = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.float32, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    if missing:
        raise SystemExit(f"The base model's weights did not load ({len(missing)} missing, e.g. {missing[:3]}). "
                         "Send this error.")
    print("Loaded the base model correctly.\n")
    return tok, model


def letter_token_ids(tok):
    """One token id per answer letter (" A", " B", ...), checked to be single tokens."""
    ids = []
    for letter in LETTERS:
        pieces = tok.encode(" " + letter, add_special_tokens=False)
        if len(pieces) != 1:
            pieces = tok.encode(letter, add_special_tokens=False)
        if len(pieces) != 1:
            raise SystemExit(f"Answer letter {letter!r} is not a single token.")
        ids.append(pieces[0])
    return torch.tensor(ids, device=DEVICE)


def option_logits(model, tok, message, letter_ids):
    """Scores for the six answer letters, read at the last position of the prompt only."""
    enc = tok(build_prompt(message), return_tensors="pt").to(DEVICE)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], logits_to_keep=1)
    return out.logits[0, -1, letter_ids]  # (6,)


@torch.no_grad()
def evaluate(model, tok, rows, letter_ids, tag):
    model.eval()
    results = []
    for row in rows:
        start = time.perf_counter()
        probs = F.softmax(option_logits(model, tok, row["text"], letter_ids), dim=-1).cpu()
        ms = (time.perf_counter() - start) * 1000
        best = int(probs.argmax())
        results.append({
            "id": row["id"], "text": row["text"], "difficulty": row["difficulty"],
            "true_label": row["label"], "predicted_label": LABEL_NAMES[best],
            "confidence": round(float(probs[best]), 3),
            "correct": LABEL_NAMES[best] == row["label"], "ms": round(ms, 1), "stage": tag,
        })
    return results


def summarize(results, title):
    n = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"=== {title} ===")
    print(f"Overall accuracy: {correct}/{n} = {correct / n:.0%}")
    right = [r["confidence"] for r in results if r["correct"]]
    wrong = [r["confidence"] for r in results if not r["correct"]]
    if right:
        print(f"Average confidence when right: {sum(right) / len(right):.2f}")
    if wrong:
        print(f"Average confidence when wrong: {sum(wrong) / len(wrong):.2f}")
    times = sorted(r["ms"] for r in results)
    print(f"Median time per message: {times[len(times) // 2]:.0f} ms")

    print("Accuracy per tutor label:")
    by_label = defaultdict(list)
    for r in results:
        by_label[r["true_label"]].append(r["correct"])
    for label, values in by_label.items():
        print(f"  {label:<11} {sum(values)}/{len(values)}")

    print("Accuracy per message type:")
    by_type = defaultdict(list)
    for r in results:
        by_type[r["difficulty"]].append(r["correct"])
    for kind, values in by_type.items():
        print(f"  {kind:<11} {sum(values)}/{len(values)}")

    print("Most common mistakes (true label -> predicted label):")
    mistakes = Counter((r["true_label"], r["predicted_label"]) for r in results if not r["correct"])
    for (true, pred), count in mistakes.most_common(5):
        print(f"  {true} -> {pred}: {count}")
    print()


def main():
    train_rows, test_rows = load_rows("train"), load_rows("test")
    print(f"Device: {DEVICE}. {len(train_rows)} training and {len(test_rows)} test messages.\n")

    tok, model = load_model()
    model.to(DEVICE)
    letter_ids = letter_token_ids(tok)

    # Before training
    before = evaluate(model, tok, test_rows, letter_ids, "before")
    summarize(before, "BEFORE training (Jeff base, no adapter)")

    # Add the LoRA adapter: only these small extra weights are trained.
    config = LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05, target_modules=LORA_TARGETS)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    targets = {name: i for i, name in enumerate(LABEL_NAMES)}

    # Train: the loss compares the six letter scores with the correct letter, like Jeff's readout over the option letters.
    total_steps = EPOCHS * len(train_rows)
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_rows)
        total = 0.0
        optimizer.zero_grad()
        for step, row in enumerate(train_rows, start=1):
            logits = option_logits(model, tok, row["text"], letter_ids)
            label = torch.tensor(targets[row["label"]], device=DEVICE)
            loss = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
            (loss / ACCUMULATE).backward()
            total += loss.item()
            if step % ACCUMULATE == 0 or step == len(train_rows):
                optimizer.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                if DEVICE == "mps":
                    torch.mps.empty_cache()  # free unused GPU memory so training does not slow down
                done = (epoch - 1) * len(train_rows) + step
                per_step = (time.perf_counter() - train_start) / done
                left_min = per_step * (total_steps - done) / 60
                print(f"  epoch {epoch}, message {step}/{len(train_rows)}: "
                      f"{per_step:.1f} s per message, about {left_min:.0f} min left", flush=True)
        print(f"Epoch {epoch}/{EPOCHS}: training loss {total / len(train_rows):.3f}", flush=True)
    print()

    # After training
    after = evaluate(model, tok, test_rows, letter_ids, "after")
    summarize(after, "AFTER training (LoRA adapter)")

    model.save_pretrained(ADAPTER_DIR)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=before[0].keys())
        writer.writeheader()
        writer.writerows(before + after)
    print(f"Adapter saved to {ADAPTER_DIR}/, predictions saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
