import os


def is_tracing_enabled() -> bool:
    """True when both a LangSmith API key and tracing flag are set."""
    has_key = bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
    is_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    return has_key and is_on


def get_project_url() -> str:
    """Return a direct link to the LangSmith project trace list."""
    project = os.getenv("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "agentcorp"))
    return f"https://smith.langchain.com/projects/p/{project}"


def configure_tracing() -> None:
    """
    Sync LANGSMITH_API_KEY → LANGCHAIN_API_KEY so LangGraph's built-in
    LangSmith integration picks it up regardless of which key name the user set.
    Also ensures LANGCHAIN_PROJECT defaults to 'agentcorp'.
    """
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "agentcorp"
