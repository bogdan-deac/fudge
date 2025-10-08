from typing import Any
from openai import AzureOpenAI

class WordExtractor:
    def __init__(self, client: AzureOpenAI, device: str, model: Any, openai_model: str):
        self.client = client
        self.device = device
        self.model = model
        self.openai_model = openai_model
        self.obscene_words = []

    def extract_words(self,result: dict):
        for segment in result['segments']:
            print(f"Segment: {segment['text']}")
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": (
                        "Find all obscene and bad words in the text (any language). "
                        "Return ONLY a valid Python list on a single line, no code blocks or markdown formatting."
                    )},
                    {"role": "user", "content": segment['text']}
                ],
                temperature=0.0,
                n=1
            )
            if response.choices[0].message.content.strip() != "[]":
                self.obscene_words.extend(eval(response.choices[0].message.content.strip()))
            print("Obscene words found:", response.choices[0].message.content.strip())
            