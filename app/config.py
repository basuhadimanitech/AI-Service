import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    embed_model: str = os.environ.get("AI_EMBED_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("AI_CHAT_MODEL", "gpt-4o-mini")

    data_dir: Path = Path(os.environ.get("AI_DATA_DIR", str(Path(__file__).parent.parent / "data")))

    # Dev CORS default - restrict to the real CPRuntime origin(s) in
    # production via the AI_ALLOWED_ORIGINS env var (comma-separated).
    allowed_origins: list[str] = (
        os.environ.get("AI_ALLOWED_ORIGINS", "*").split(",")
    )

    top_k_chunks: int = 5


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
