
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Project root: the directory containing the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using local environment configuration."""

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add your API key to the .env file."
        )

    return OpenAI(api_key=OPENAI_API_KEY)
