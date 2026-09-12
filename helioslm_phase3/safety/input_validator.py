"""Input validation and sanitization."""
import re
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    sanitized_input: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class InputValidator:
    """
    Validate and sanitize API inputs.

    Checks:
      - Max prompt length
      - Max tokens
      - Valid UTF-8
      - No control characters
      - Reasonable temperature/top_p values
    """

    def __init__(
        self,
        max_prompt_length: int = 100000,
        max_tokens_limit: int = 8192,
        max_batch_size: int = 32,
    ):
        self.max_prompt_length = max_prompt_length
        self.max_tokens_limit = max_tokens_limit
        self.max_batch_size = max_batch_size

    def validate_chat_request(self, request: dict) -> ValidationResult:
        """Validate chat completion request."""
        errors = []

        # Check messages
        messages = request.get("messages", [])
        if not messages:
            errors.append("messages field is required")

        if len(messages) > self.max_batch_size:
            errors.append(f"Too many messages: {len(messages)} > {self.max_batch_size}")

        # Check each message
        total_length = 0
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"Message {i} must be an object")
                continue

            role = msg.get("role", "")
            if role not in ["system", "user", "assistant", "tool"]:
                errors.append(f"Invalid role: {role}")

            content = msg.get("content", "")
            if not isinstance(content, str):
                errors.append(f"Message {i} content must be a string")

            total_length += len(content)

        if total_length > self.max_prompt_length:
            errors.append(f"Prompt too long: {total_length} > {self.max_prompt_length}")

        # Check parameters
        max_tokens = request.get("max_tokens", 256)
        if max_tokens > self.max_tokens_limit:
            errors.append(f"max_tokens too large: {max_tokens} > {self.max_tokens_limit}")

        temperature = request.get("temperature", 0.7)
        if not (0 <= temperature <= 2):
            errors.append(f"temperature must be between 0 and 2: {temperature}")

        top_p = request.get("top_p", 0.9)
        if not (0 <= top_p <= 1):
            errors.append(f"top_p must be between 0 and 1: {top_p}")

        # Sanitize
        sanitized = self._sanitize_messages(messages)

        return ValidationResult(
            valid=len(errors) == 0,
            sanitized_input=sanitized,
            errors=errors,
        )

    def _sanitize_messages(self, messages: List[dict]) -> str:
        """Sanitize and format messages."""
        sanitized = []
        for msg in messages:
            content = msg.get("content", "")
            # Remove control characters
            content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', content)
            # Normalize whitespace
            content = ' '.join(content.split())
            sanitized.append({**msg, "content": content})
        return sanitized
