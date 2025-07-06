from data_loader import DataLoader
import whisper

data_loader = DataLoader("../video resources/v4.mp4")
data = data_loader.load_data()

model = whisper.load_model("large-v3")
result = model.transcribe(data.get("audio_path"), language="ro")
print(result["text"])
print("Detected language:", result.get("language", "unknown"))