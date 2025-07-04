import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import assemblyai as aai

load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not assemblyai_api_key or not openai_api_key:
    raise ValueError("API keys not properly set in .env")

aai.settings.api_key = assemblyai_api_key
openai_client = AsyncOpenAI(api_key=openai_api_key)