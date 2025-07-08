from dotenv import load_dotenv
import os


if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
    API_VERSION = os.getenv('API_VERSION')
    # TODO - integrate with STT