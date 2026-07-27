import os
from pathlib import Path


class Settings:
    # Ollama now runs on a shared host inside the Adobe network rather than
    # on the learner's machine - content still never leaves the Adobe
    # environment, it just no longer requires a per-learner local install.
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.environ.get("AI_EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.environ.get("AI_CHAT_MODEL", "llama3.2")
    vision_model: str = os.environ.get("AI_VISION_MODEL", "llava")

    data_dir: Path = Path(os.environ.get("AI_DATA_DIR", str(Path(__file__).parent.parent / "data")))

    # Dev CORS default - restrict to the real CPRuntime origin(s) in
    # production via the AI_ALLOWED_ORIGINS env var (comma-separated).
    allowed_origins: list[str] = (
        os.environ.get("AI_ALLOWED_ORIGINS", "*").split(",")
    )

    top_k_chunks: int = 5


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
