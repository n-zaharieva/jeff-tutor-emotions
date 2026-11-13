import jeff

client = jeff.Jeff.local(["emotion"])

emotion = client.function("jeff.language.emotion.emotion", context={
    "source": "A student's message to an online tutor",
})

messages = [
    "I've tried this 4 times and still get it wrong. I'm so stupid.",
    "Oh great, more fractions. My favourite.",
    "I finally understand derivatives, thank you!",
    "I don't get what the question is asking.",
]

for m in messages:
    d = emotion(m)
    print(f"{m}\n  -> {d.value} (confidence {d.confidence:.2f}, unsure: {d.abstained})\n")
