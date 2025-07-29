from dotenv import load_dotenv
import os
from STT.data_loader import DataLoader
from STT.word_extraction import WordExtractor
from openai import AzureOpenAI
import whisperx
import argparse
import torch




if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
    API_VERSION = os.getenv('API_VERSION')
    WHISPER_MODEL = os.getenv('WHISPER_MODEL')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL')

    parser = argparse.ArgumentParser(description="A simple CLI example")
    parser.add_argument("--video_path", help="the path of the video")

    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = DataLoader.load_data(file_path=video_path)

    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT
    )

    whisper_model = whisperx.load_model(WHISPER_MODEL, device=device, compute_type="int8")


    extractor = WordExtractor(client=client, device=device, model=whisper_model, openai_model=OPENAI_MODEL)
    extractor.extract_words(video_path=video_path, data=data)
