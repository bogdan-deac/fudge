from typing import Any
from openai import AzureOpenAI
import whisperx


class WordExtractor:
    def __init__(self, client: AzureOpenAI, device: str, model: Any, openai_model: str):
        self.client = client
        self.device = device
        self.model = model
        self.openai_model = openai_model

    def extract_words(self,video_path: str, data: dict[str, Any]): 
        result = self.model.transcribe(data.get("audio_path"), language="ro")
        print(result)
        print("Detected language:", result.get("language", "unknown"))

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        aligned_result = whisperx.align(result["segments"], model_a, metadata, video_path, device=self.device)

        print("Aligned segments:", aligned_result["word_segments"])

        for word in aligned_result["word_segments"]:
            if all(k in word for k in ("word", "start", "end")):
                print(f"{word['word']} => {word['start']} - {word['end']}")
            else:
                print("Missing or incomplete segment:", word)
                
        for segment in result['segments']:
            print(f"Segment: {segment['text']}")
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Find all obscene and bad words in the text (no matter the language the text is in). Return ONLY a valid Python list (on a single line), WITHOUT using code blocks or markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": segment['text']
                    },
                ],
                temperature=0.0,
                n=1
            )
            print("Obscene words found:", response.choices[0].message.content.strip())
            print("-" * 40)