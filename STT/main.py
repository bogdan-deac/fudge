from data_loader import DataLoader
import whisperx
import torch

video_path = "../video resources/v1.mp4"
data_loader = DataLoader(video_path)
data = data_loader.load_data()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v3", device=device, compute_type="int8")
result = model.transcribe(data.get("audio_path"), language="ro")
print(result)
print("Detected language:", result.get("language", "unknown"))

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
aligned_result = whisperx.align(result["segments"], model_a, metadata, video_path, device="cpu")

print("Aligned segments:", aligned_result["word_segments"])

for word in aligned_result["word_segments"]:
    if all(k in word for k in ("text", "start", "end")):
        print(f"{word['text']} => {word['start']} - {word['end']}")
    else:
        print("Missing or incomplete segment:", word)