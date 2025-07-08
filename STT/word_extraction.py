from config import CONFIG
from openai import AzureOpenAI
from data_loader import DataLoader
import whisperx
import torch

client = AzureOpenAI(
    api_key=CONFIG["OPENAI_API_KEY"],
    api_version=CONFIG["API_VERSION"],
    azure_endpoint=CONFIG["AZURE_ENDPOINT"]
)

video_path = "../video resources/v4.mp4"
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
    if all(k in word for k in ("word", "start", "end")):
        print(f"{word['word']} => {word['start']} - {word['end']}")
    else:
        print("Missing or incomplete segment:", word)
        
for segment in result['segments']:
    print(f"Segment: {segment['text']}")
    response = client.chat.completions.create(
        model=CONFIG["MODEL"],
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