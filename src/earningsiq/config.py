from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    # .env is looked up in the project root, not the current directory
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    raw_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "processed")
    embeddings_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "embeddings")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")

    # Contact e-mail sent to SEC EDGAR with every download. Must be set by
    # the user (EDGAR_USER_AGENT); see .env.example.
    edgar_user_agent: str = ""
    edgar_start_year: int = 2019
    edgar_end_year: int = 2024
    edgar_filing_types: list[str] = Field(default_factory=lambda: ["10-Q", "10-K"])

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 100

    llm_model: str = "auto"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1

    vector_db_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "chromadb")
    vector_collection_name: str = "earnings_reports"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    batch_size: int = 8
    num_workers: int = 4


settings = Settings()
