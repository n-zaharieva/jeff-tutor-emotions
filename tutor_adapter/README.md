# Tutor emotion adapter (LoRA for Jeff)

The LoRA adapter trained in experiment 3 (`train_tutor_adapter.py`). It lets Jeff's base model
label a student message as frustrated, confused, bored, confident, neutral or distressed.

- **Base model:** `mstrasser/jeff-base`, revision `v1.4`
- **Training data:** the 120 training messages in `tutor_emotions.csv`
- **Settings:** LoRA rank 16, alpha 32, dropout 0.05, top 8 of 24 text layers (16–23), 3 epochs, learning rate 2e-4
- **Result:** 72% accuracy on the 60 test messages (57% before training); see the main README

## Files

| File | What it is |
|---|---|
| `adapter_config.json` | The LoRA settings and the base model it belongs to |
| `adapter_model.safetensors` | The trained weights (about 3.4 million) |

## Loading it

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = AutoModelForImageTextToText.from_pretrained("mstrasser/jeff-base", revision="v1.4")
tokenizer = AutoTokenizer.from_pretrained("mstrasser/jeff-base", revision="v1.4")
model = PeftModel.from_pretrained(base, "tutor_adapter")
```

The prompt format and how the answer is read from the option letters are in `train_tutor_adapter.py`
(`build_prompt` and `option_logits`).
