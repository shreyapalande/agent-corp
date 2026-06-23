import time
import google.generativeai as genai

from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL = "gemini-2.5-flash"


def _get_api_keys() -> list[str]:
    import os
    # Collect GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... sorted numerically
    numbered = sorted(
        ((int(k[len("GEMINI_API_KEY_"):]), v) for k, v in os.environ.items()
         if k.startswith("GEMINI_API_KEY_") and k[len("GEMINI_API_KEY_"):].isdigit()),
        key=lambda x: x[0],
    )
    keys = [v for _, v in numbered if v.strip()]
    # Fall back to plain GEMINI_API_KEY if no numbered keys found
    if not keys:
        fallback = os.environ.get("GEMINI_API_KEY", "").strip()
        if fallback:
            keys = [fallback]
    if not keys:
        raise ValueError(
            "No Gemini API key configured. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... in .env"
        )
    return keys


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg


def call_gemini(prompt: str, *, temperature: float, max_output_tokens: int) -> genai.types.GenerateContentResponse:
    """
    Call Gemini Flash 2.5, rotating to the next API key on rate-limit errors.
    Raises RuntimeError if all keys are exhausted.
    """
    keys = _get_api_keys()
    last_exc: Exception | None = None

    for i, key in enumerate(keys):
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                _MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            response = model.generate_content(prompt)
            if i > 0:
                logger.info("gemini_client | rate-limit fallback succeeded on key index %d", i)
            return response
        except Exception as exc:
            if _is_rate_limit(exc):
                logger.warning(
                    "gemini_client | rate limit on key index %d/%d, trying next | error=%s",
                    i + 1, len(keys), exc,
                )
                last_exc = exc
            else:
                raise

    raise RuntimeError(
        f"All {len(keys)} Gemini API key(s) hit rate limits. Last error: {last_exc}"
    ) from last_exc
