from typing import Any
from openai import AzureOpenAI

class WordSubstitutor:
    def __init__(self, client: AzureOpenAI, device: str, model: Any, openai_model: str, obscene_words: list[str], language: str):
        self.client = client
        self.device = device
        self.model = model
        self.openai_model = openai_model
        self.obscene_words = obscene_words
        self.language = language

    def substitute_words(self):
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": (
                    "Replace all obscene or offensive words in the given list with neutral, child-friendly replacements. "
                    "Each replacement must be a **real word** in the **same language** as the original, similar in structure, "
                    "syllable count, or rhythm for easy audio substitution, and preserve general sound if possible. "
                    "While some words may have multiple meanings, focus on the most common or widely understood meaning. "
                    "Whisper detected the language as: " + self.language + ". "
                    "Return ONLY a valid Python list on a single line, one replacement per word, in the same order, with no "
                    "explanations or formatting.")},
                {"role": "user", "content": str(self.obscene_words)}
            ],
            temperature=0.0,
            n=1
        )
        print("Obscene words replacements:", response.choices[0].message.content.strip())