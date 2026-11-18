# Reading student emotions with Jeff: an emotion-aware layer for AI tutors

 This repository is about an emotion-reading layer that sits in front of a tutor. For every student message
it decides, in a fraction of a second, how the student is doing, so the tutor can adapt its response or hand
the student over to a person.

It is built on **Jeff**, a very small open-source "decision model" that does not write text. Instead of a chat
reply, it returns one choice from a list of options together with a probability.

---

## Contents

1. [What Jeff is and why I chose it](#1-what-jeff-is-and-why-i-chose-it)
2. [Application and design](#2-application-and-design)
3. [Data](#3-data)
4. [Experiments](#4-experiments)
5. [Results](#5-results)
6. [How to run it](#6-how-to-run-it)
7. [Files](#7-files)
8. [Limitations and ethics](#8-limitations-and-ethics)

---

## 1. What Jeff is and why I chose it

[Jeff](https://github.com/firelex/jeff) is an open-source "System 1" model of about 0.8 billion parameters,
fine-tuned from Qwen3.5-0.8B. You describe a situation and a list of options in plain words, Jeff returns a
probability for each option in a single forward pass, with no generated text. Small add-ons called
adapters (LoRA weights) specialise it for one task each.

Why Jeff for this application:

- **Speed** - an emotion check has to run on every message before the tutor replies. Jeff answers in about
  0.15 to 0.4 seconds on a laptop.
- **A decision, not a paragraph** - the tutor's code needs one label ("distressed"), not text to parse.
- **A probability with every answer** - this allows a simple rule: act when the model is sure, ask a human when
  it is not
- **Open source and local** - code under MIT, weights under Apache 2.0. It runs on a laptop, so student
  messages never leave the device.
- **New** - Jeff was released in autumn 2026, so testing it in an educational, affective setting is new ground.

The contrast with my group members' work: their tutors generate explanations, Jeff decides how the student
feels, which is the affective computing side of the project.

## 2. Application and design

The agent is not embodied. It is an invisible layer inside a text chatbot: before the tutor
replies, Jeff reads the student's message and decides how the student is doing. No avatar or robot is needed to
read emotions from text, and because the layer has no face of its own, it can sit behind any tutor, such as
the calculus and Dutch tutors of my group members.

The only input is text: the message the student types. This matches how students already use a
tutor chatbot, so nothing extra is asked of them. It is also a limitation: Jeff reads English text only, so tone
of voice, facial expressions and typing behaviour, which could reveal frustration or distress, are not available.

The direct users are university students who use an AI tutor for calculus or Dutch. They do
not see Jeff, only a tutor that adapts to them. Teachers and student counsellors are indirect users: they receive
the cases Jeff flags, such as a possibly distressed student, because these decisions should not be left to the
model alone.

The six labels and what the tutor does with each:

| Label | Tutor's response |
|---|---|
| frustrated | encourage |
| confused | explain again |
| bored | make it more engaging |
| confident | move on to a harder task |
| neutral | continue normally |
| distressed | hand over to a human (teacher or counsellor) |

## 3. Data

`tutor_emotions.csv` contains 180 short student messages about calculus and Dutch, 30 per label.

- 120 training messages (20 per label) and 60 test messages (10 per label). The test messages are never
  used for training.
- The test set deliberately includes hard cases, marked in the `difficulty` column: `sarcasm`, `short`
  ("Huh?", "ok..."), `implicit` (feelings not stated directly), `mixed` (two feelings at once) and `plain`.
- The messages were written with the help of an AI assistant and checked and labelled by me. They are not
  real student data.

## 4. Experiments

All experiments are scored on the **same 60 test messages**.

| # | Script | Approach | Training? |
|---|---|---|---|
| 1 | `evaluate_baseline.py` | Jeff's ready-made emotion adapter (27 emotions plus neutral, trained on Reddit comments). Its emotions are mapped onto the six tutor labels (for example annoyance to frustrated, fear to distressed) | No |
| 2 | `evaluate_described_labels.py` | Jeff's triage adapter, which normally routes support emails to teams. The six tutor labels are given as "teams", each with a one-line description (prompt engineering) | No |
| 3a | `train_tutor_adapter.py` (before) | Jeff's base model with a prompt listing the labels as options A to F; the answer is read from the probabilities of those letters | No |
| 3b | `train_tutor_adapter.py` (after) | The same, after fine-tuning a LoRA adapter on the 120 training messages | Yes |


**Fine-tuning details (3b)**
Jeff's own trainer is not published, so I used standard LoRA from Hugging Face
PEFT, keeping Jeff's core idea: the loss is a cross-entropy over the six option letters at the last position of
the prompt. 
Settings: LoRA rank 16 (the same rank as Jeff's own adapters), alpha 32, dropout 0.05, learning
rate 2e-4, 3 epochs, one message per step with 4 accumulated steps per update, seed 42. LoRA was applied only
to the top 8 of the 24 text layers (16 to 23). Training all layers ran out of memory and took over
15 seconds per message on a laptop, while this setup took about 1 second per message (about 8 minutes in
total).

## 5. Results

Accuracy on the 60 test messages (10 per label):

| | 1. Emotion adapter | 2. Described labels | 3a. Base, letter prompt | **3b. Fine-tuned** |
|---|---|---|---|---|
| **Overall** | 47% | 57% | 57% | **72%** |
| frustrated | 4/10 | 5/10 | 5/10 | **8/10** |
| confused | 5/10 | 6/10 | 3/10 | **8/10** |
| bored | 0/10 | 7/10 | 2/10 | **8/10** |
| confident | 7/10 | 9/10 | **10/10** | **10/10** |
| neutral | **9/10** | 2/10 | 7/10 | 4/10 |
| distressed | 3/10 | 5/10 | **7/10** | 5/10 |
| sarcasm (4) | 0/4 | 1/4 | 0/4 | **2/4** |
| implicit (9) | 2/9 | 2/9 | 3/9 | **5/9** |
| Median time per message | 294 ms | 153 ms | 271 ms | 375 ms |

Confidence: in experiment 1 the model counted itself sure on only 5 of 60 messages; in experiment 2 on 58 of 60,
while being right only 57% of the time. In 3a the average confidence was 0.71 when right and 0.59 when wrong;
in 3b, 0.86 when right and 0.76 when wrong.

**Main findings**

1. **Fine-tuning helped most:** 57% to 72% with only 120 examples and minutes of laptop training.
2. **Labels matter:** the ready-made emotion adapter has no "bored" emotion and could never detect it.
3. **Sarcasm and hidden feelings are the weak spots of every version.**
4. **Safety trade-off:** fine-tuning raised overall accuracy but lowered recognition of distressed students
   (7/10 to 5/10). Three distressed students were labelled "frustrated" and would have been encouraged instead
   of being referred to a person.
5. **Confidence does not transfer:** adapters calibrated on other tasks (emails, Reddit comments) were either
   almost never sure or almost always sure on tutoring messages, and even the fine-tuned model was quite
   confident when wrong.

All per-message predictions are in `results_baseline.csv`, `results_described_labels.csv` and
`results_finetuned.csv` (the last one contains both the "before" and "after" rows, marked in the `stage` column).

## 6. How to run it

Tested on macOS with Apple silicon. Jeff's local mode runs on Apple silicon or NVIDIA GPUs.

**1. Install uv** (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # if uv is not found afterwards
```

**2. Install Jeff next to this repository:**

```bash
cd ~
git clone https://github.com/firelex/jeff.git
cd jeff
uv python pin 3.14     # PyTorch has no wheels for Python 3.15 yet
uv sync
```

Jeff version used: *59898ae*, base model `mstrasser/jeff-base`
revision `v1.4`.

**3. Run the experiments from this repository's folder**, using Jeff's environment (`--project ../jeff`):

```bash
cd ~/jeff-tutor-emotions

# Experiment 1: ready-made emotion adapter
uv run --project ../jeff python evaluate_baseline.py

# Experiment 2: tutor labels described in words (triage adapter)
uv run --project ../jeff python evaluate_described_labels.py

# Experiment 3: base model before and after LoRA fine-tuning
uv run --project ../jeff --with peft --with transformers --with accelerate python train_tutor_adapter.py
```

The first run of each script downloads the model and adapters from Hugging Face (about 1.8 GB). Warnings about
`causal_conv1d` and `flash-linear-attention` can be ignored: these speed-ups exist only for NVIDIA GPUs.

## 7. Files

| File | What it is |
|---|---|
| `tutor_emotions.csv` | The 180 labelled messages (`id`, `text`, `label`, `split`, `difficulty`) |
| `evaluate_baseline.py` | Experiment 1 |
| `evaluate_described_labels.py` | Experiment 2 |
| `train_tutor_adapter.py` | Experiment 3 (evaluation before training, LoRA training, evaluation after) |
| `results_*.csv` | Every prediction of every experiment |
| `tutor_adapter/` | The trained LoRA adapter (PEFT format) |
| `test_jeff.py`, `test_emotion.py` | First checks that Jeff runs |

## 8. Limitations and ethics

**Limitations**

- Small test set: 60 messages, so one message is about 1.7 percentage points; differences of one or two
  messages per label may be chance.
- One training run with one random seed.
- AI-assisted, not real, data; real students write differently (typos, other languages, longer context).
- Text only and one message at a time: the model does not see the conversation history.
- English only: the Dutch tutor's students may write in Dutch.
- The label mapping in experiment 1 was my own choice; another mapping would change its results.

### Human-AI teaming

Jeff should not act alone on emotions. In deployment it would sort messages so that the tutor adapts its tone
for clear cases, while possible distress is always routed to a teacher or counsellor, and uncertain decisions
are reviewed by a person. The results show why a person must stay in the loop: even the best model missed half
of the distressed students and was often confident when wrong. A deployed system would also need a stricter
rule for distress (escalate even at low probability), regular checks of its mistakes by staff, and a way for
students to reach a person directly.

### Ethics

- Privacy and surveillance: reading students' emotions from their messages is sensitive. Students should
  know it happens and be able to opt out. Running Jeff locally helps, since messages don't need to be sent elsewhere.
- Missed distress: a false "frustrated" for a distressed student can delay real help.
- False confidence: a precise-looking probability can make a wrong decision look trustworthy.
- Bias: the training messages were written in one style. Students who express themselves differently
  (non-native speakers, other cultures, neurodivergent students) may be misread more often.
- AI-generated data: the dataset was written with AI assistance, which may not reflect how real students talk.
