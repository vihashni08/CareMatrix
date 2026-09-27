"""Shared LLM client, exceptions, and base reasoning engine for CareMatrix.

Provides a unified interface to Google Gemini across all reasoning agents:
- Standardized API key lookup (parameter -> env -> .env -> fallback)
- Lazy client initialization
- Resilient model fallback and timeout handling
- Clean JSON stripping and extraction
- Standardized exception hierarchy
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

logger = logging.getLogger("CareMatrix.LLMClient")


class LLMReasonerError(Exception):
    """Base exception for LLM reasoning failures."""


class LLMUnavailableError(LLMReasonerError):
    """Raised when Gemini client or API key is not configured."""


class LLMSchemaValidationError(LLMReasonerError):
    """Raised when Gemini response fails structured schema validation."""


class LLMTimeoutError(LLMReasonerError):
    """Raised when Gemini request exceeds the allocated timeout."""


def clean_json_text(text: str) -> str:
    """Strip markdown code blocks or extraneous formatting from model response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def find_api_key(explicit_key: str | None = None) -> str | None:
    """Find GEMINI_API_KEY from parameter, environment, or .env file."""
    if explicit_key:
        return explicit_key
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY")

    search_dirs = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
        Path.home(),
    ]
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
            except Exception:
                pass

    return ""


class BaseGeminiReasoner:
    """Abstract base class for Gemini-backed agent reasoners with fallback and validation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = find_api_key(api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self.timeout = timeout
        self._client = client

    @property
    def is_available(self) -> bool:
        """Check if Gemini client is initialized or API key is present."""
        return self._client is not None or bool(self.api_key)

    def _get_client(self) -> Any:
        """Lazily initialize Google Gemini Client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise LLMUnavailableError("GEMINI_API_KEY is not configured.")

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception as exc:
            raise LLMUnavailableError(f"Failed to initialize Gemini Client: {exc}") from exc

    def call_gemini(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Invoke Gemini with resilience retry, model fallback pool, and timeout detection."""
        client = self._get_client()

        candidate_pool = [
            "gemini-flash-lite-latest",
            self.model_name,
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
        ]
        models_to_try = []
        for m in candidate_pool:
            if m and m not in models_to_try:
                models_to_try.append(m)

        raw_text = None
        last_error = None

        for current_model in models_to_try:
            for attempt in range(2):
                try:
                    if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
                        interaction = client.interactions.create(
                            model=current_model,
                            input=prompt,
                            system_instruction=system_instruction,
                        )
                        raw_text = getattr(interaction, "output_text", None) or getattr(interaction, "text", None)
                    elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
                        config = None
                        try:
                            from google.genai import types
                            cfg_kwargs: dict[str, Any] = {"response_mime_type": "application/json"}
                            if system_instruction:
                                cfg_kwargs["system_instruction"] = system_instruction
                            config = types.GenerateContentConfig(**cfg_kwargs)
                        except Exception:
                            config = None

                        response = client.models.generate_content(
                            model=current_model,
                            contents=prompt,
                            config=config,
                        )
                        raw_text = getattr(response, "text", None)
                    elif hasattr(client, "generate_content"):
                        response = client.generate_content(prompt)
                        raw_text = getattr(response, "text", None)
                    else:
                        raise LLMUnavailableError("Gemini client does not have a supported generation interface.")

                    if raw_text:
                        break
                    raise LLMSchemaValidationError("Gemini returned empty response text.")

                except (LLMUnavailableError, LLMSchemaValidationError):
                    raise
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()
                    if "timeout" in err_str or "timed out" in err_str:
                        raise LLMTimeoutError(f"Gemini reasoning timed out: {exc}") from exc
                    if any(k in err_str for k in ("429", "resource_exhausted", "quota", "503", "unavailable", "404", "not_found")):
                        time.sleep(0.5)
                        break
                    raise LLMReasonerError(f"Gemini API request failed: {exc}") from exc

            if raw_text:
                break

        if not raw_text:
            raise LLMReasonerError(f"Gemini API request failed: {last_error}")

        return raw_text


__all__ = [
    "BaseGeminiReasoner",
    "LLMReasonerError",
    "LLMSchemaValidationError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "clean_json_text",
    "find_api_key",
]
