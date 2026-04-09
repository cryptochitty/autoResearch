import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "10"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/outputs")

if not ANTHROPIC_API_KEY:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
