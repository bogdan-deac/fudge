import os
import whisperx
import argparse
import torch
from dotenv import load_dotenv
from openai import AzureOpenAI
from torch.serialization import add_safe_globals
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig

from STT.data_loader import DataLoader
from STT.word_extraction import WordExtractor
from STT.obscene_word_substitution import WordSubstitutor
from STT.word_aligner import WordAligner

if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
    API_VERSION = os.getenv('API_VERSION')
    WHISPER_MODEL = os.getenv('WHISPER_MODEL')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL')
    XTTS_MODEL_PATH = os.getenv('XTTS_MODEL_PATH')

    parser = argparse.ArgumentParser(description="A simple CLI example")
    parser.add_argument("--video_path", help="the path of the video")

    args = parser.parse_args()

    video_path = os.path.abspath(args.video_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    data = DataLoader.load_data(file_path=video_path)

    client = AzureOpenAI(        
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
    )

    whisper_model = whisperx.load_model(WHISPER_MODEL, device=device, compute_type="int8")

    aligner = WordAligner(device=device, model=whisper_model, data=data, video_path=video_path)
    aligner.align_words()
    result = aligner.result
    language = aligner.language
    print("Finished aligning words.")

    extractor = WordExtractor(client=client, device=device, model=whisper_model, openai_model=OPENAI_MODEL)
    extractor.extract_words(result=result)
    print("Finished extracting obscene words.")

    substitutor = WordSubstitutor(client=client, device=device, model=whisper_model, openai_model=OPENAI_MODEL, obscene_words=extractor.obscene_words, language=language)
    substitutor.substitute_words()
    print("Finished substituting obscene words.")

    for i, segment in enumerate(result['segments']):
        text = segment['text']
        for word, replacement in zip(substitutor.obscene_words, substitutor.obscene_words_replacements):
            text = text.replace(word, replacement)
        segment['text'] = text
        print(f"Segment {i} after replacement: {segment['text']}")

    add_safe_globals([XttsConfig, XttsAudioConfig])
    tts = TTS(XTTS_MODEL_PATH).to(device)
    tts.tts_to_file(
        text=result['segments'][0]['text'],
        speaker_wav=os.path.splitext(video_path)[0] + "_audio.wav",
        language=language if language != "unknown" else "en",
        file_path=os.path.splitext(video_path)[0] + "_cloned_replacement.wav",
        speed=1.5,
        split_sentences=False
    )
    print("Finished generating replacement audio.")