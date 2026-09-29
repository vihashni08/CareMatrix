"""Unit tests for Gemini model selection, fallback pool, and timeout handling.

Verifies:
  - BaseGeminiReasoner calls the primary model when available.
  - When primary model returns 404/429/503, reasoner falls back to candidate pool models.
  - Timeout errors raise LLMTimeoutError.
  - Empty response text raises LLMSchemaValidationError.
  - Missing API key raises LLMUnavailableError.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import unittest

from communication.llm_client import (
    BaseGeminiReasoner,
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
)


class TestGeminiModelSelection(unittest.TestCase):
    """Test suite verifying client resilience, model fallback pool, and timeout detection."""

    def test_missing_api_key_raises_unavailable(self):
        """Without API key or client, call_gemini must raise LLMUnavailableError."""
        reasoner = BaseGeminiReasoner(client=None)
        reasoner.api_key = ""  # explicitly clear key
        with self.assertRaises(LLMUnavailableError):
            reasoner.call_gemini("Test prompt")

    def test_successful_primary_model_call(self):
        """When client succeeds on primary model, return text directly."""
        mock_client = MagicMock(spec=["models"])
        mock_response = MagicMock()
        mock_response.text = '{"decision": "ESCALATE", "rationale": "High risk"}'
        mock_client.models.generate_content.return_value = mock_response

        reasoner = BaseGeminiReasoner(client=mock_client)
        output = reasoner.call_gemini("Evaluate patient telemetry")
        self.assertIn("ESCALATE", output)
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_model_fallback_on_404_or_quota_exhaustion(self):
        """When first model fails with 404/quota, fallback to subsequent models in pool."""
        mock_client = MagicMock(spec=["models"])
        mock_response = MagicMock()
        mock_response.text = '{"decision": "CONFIRM"}'

        calls = []

        def side_effect(model, contents, config=None):
            calls.append(model)
            if len(calls) == 1:
                raise Exception("404 Model Not Found: model is deprecated")
            return mock_response

        mock_client.models.generate_content.side_effect = side_effect

        reasoner = BaseGeminiReasoner(client=mock_client)
        output = reasoner.call_gemini("Evaluate telemetry")
        self.assertEqual(output, '{"decision": "CONFIRM"}')
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])

    def test_timeout_raises_llm_timeout_error(self):
        """When Gemini request times out, raise LLMTimeoutError."""
        mock_client = MagicMock(spec=["models"])
        mock_client.models.generate_content.side_effect = Exception("Request timed out after 30.0s")

        reasoner = BaseGeminiReasoner(client=mock_client)
        with self.assertRaises(LLMTimeoutError):
            reasoner.call_gemini("Evaluate telemetry")

    def test_empty_response_raises_schema_validation_error(self):
        """When Gemini returns an empty or whitespace text, raise LLMSchemaValidationError."""
        mock_client = MagicMock(spec=["models"])
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        reasoner = BaseGeminiReasoner(client=mock_client)
        with self.assertRaises(LLMSchemaValidationError):
            reasoner.call_gemini("Evaluate telemetry")


if __name__ == "__main__":
    unittest.main()
