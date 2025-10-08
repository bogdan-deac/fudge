from typing import Any
import whisperx

class WordAligner:
    def __init__(self, device: str, model: Any, data: dict, video_path: str):
        self.device = device
        self.model = model
        self.data = data
        self.video_path = video_path
        self.language = "unknown"
        self.result = {}
        self.aligned_result = {}

    def align_words(self) -> None:
        self.result = self.model.transcribe(self.data.get("audio_path"))
        print(self.result)
        self.language = self.result.get("language", "unknown")
        print("Detected language:", self.language)

        model_a, metadata = whisperx.load_align_model(language_code=self.result["language"], device=self.device)
        self.aligned_result = whisperx.align(self.result["segments"], model_a, metadata, self.video_path, device=self.device)
        print("Aligned segments:", self.aligned_result["word_segments"])

        for word in self.aligned_result["word_segments"]:
            if all(k in word for k in ("word", "start", "end")):
                print(f"{word['word']} => {word['start']} - {word['end']}")
            else:
                print("Missing or incomplete segment:", word)
