from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Search
    tavily_api_key: str = ""

    # LLM
    groq_api_key: str = ""

    # LangSmith tracing
    langchain_api_key: str = ""
    langsmith_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "agentcorp"

    # API server
    api_base_url: str = "http://localhost:8000"
    api_version: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
