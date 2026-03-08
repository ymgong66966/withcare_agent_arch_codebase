"""
Anthropic Claude client with Langfuse tracking support.
"""

from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Please install anthropic: pip install anthropic")

try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except Exception:
    logging.warning("Langfuse not available or not configured.")
    langfuse = None

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Base Anthropic Claude client"""

    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or passed to constructor")
        self.client = Anthropic(api_key=self.api_key)

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs,
    ) -> Tuple[str, Dict[str, int]]:
        """Call the Anthropic API and return (text, usage_dict).

        The usage dict has keys: input_tokens, output_tokens.
        """
        response = self.client.messages.create(
            model=kwargs.pop("model", self.model_id),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            **kwargs,
        )

        text = ""
        if response.content and len(response.content) > 0:
            text = response.content[0].text

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        return text, usage

    def converse(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """Synchronous conversation with Claude. Returns response text."""
        text, _ = self._call_api(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
        return text


class TrackedAnthropicClient(AnthropicClient):
    """AnthropicClient with Langfuse tracking"""

    def __init__(
        self,
        session_id: str = None,
        agent_role: str = None,
        user_id: str = None,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_id=model_id, api_key=api_key)
        self.session_id = session_id or f"anthropic-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.agent_role = agent_role or "unknown-agent"
        self.user_id = user_id or "anonymous"
        self.trace = None  # set externally to enable per-turn trace hierarchy

    def _start_generation(
        self,
        name: str,
        input_data: Any,
        model: str,
        metadata: Dict[str, Any],
        trace: Any = None,
    ):
        """Create a Langfuse generation, optionally as child of a trace/span.

        If no parent trace is provided, creates an ephemeral trace first
        (required by Langfuse SDK — generations must live under a trace).
        """
        if not langfuse:
            return None
        try:
            parent = trace
            if parent is None:
                # No per-turn trace available — create a standalone trace
                parent = langfuse.trace(
                    name=f"standalone-{name}",
                    session_id=self.session_id,
                    user_id=self.user_id,
                )
            return parent.generation(
                name=name,
                model=model,
                input=input_data,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to create Langfuse generation: {e}")
            return None

    def _end_generation(
        self,
        generation: Any,
        output: str,
        usage: Dict[str, int],
        duration_ms: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """End a Langfuse generation with output and usage."""
        if not generation:
            return
        try:
            meta = {"duration_ms": duration_ms, "model": self.model_id}
            if extra_metadata:
                meta.update(extra_metadata)
            generation.end(
                output=output,
                usage={
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                    "total": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                metadata=meta,
            )
        except Exception as e:
            logger.warning(f"Failed to end Langfuse generation: {e}")

    def _error_generation(self, generation: Any, error: Exception):
        """Log error on a Langfuse generation."""
        if not generation:
            return
        try:
            generation.end(level="ERROR", metadata={"error": str(error)})
        except Exception as e:
            logger.warning(f"Failed to log error to Langfuse: {e}")

    def simple_chat(self, prompt: str, max_tokens: int = 1000, trace: Any = None, **kwargs) -> str:
        """Simple chat with Langfuse logging"""
        generation = self._start_generation(
            name=f"anthropic-{self.agent_role}-simple-chat",
            input_data=prompt,
            model=self.model_id,
            metadata={"max_tokens": max_tokens, "agent_role": self.agent_role, **kwargs},
            trace=trace or self.trace,
        )

        try:
            start_time = datetime.now()
            messages = [{"role": "user", "content": prompt}]
            text, usage = self._call_api(messages, max_tokens=max_tokens)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self._end_generation(generation, text, usage, duration_ms)
            return text
        except Exception as e:
            self._error_generation(generation, e)
            raise

    async def async_chat(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        trace: Any = None,
        **kwargs,
    ) -> str:
        """Async chat with Langfuse logging"""
        generation = self._start_generation(
            name=f"anthropic-{self.agent_role}-async-chat",
            input_data=prompt,
            model=self.model_id,
            metadata={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "agent_role": self.agent_role,
                **kwargs,
            },
            trace=trace or self.trace,
        )

        try:
            start_time = datetime.now()
            messages = [{"role": "user", "content": prompt}]
            loop = asyncio.get_event_loop()
            text, usage = await loop.run_in_executor(
                None,
                lambda: self._call_api(
                    messages, max_tokens=max_tokens, temperature=temperature,
                ),
            )
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            self._end_generation(generation, text, usage, duration_ms)
            return text
        except Exception as e:
            self._error_generation(generation, e)
            raise

    async def create_message(
        self,
        *,
        model: str = None,
        max_tokens: int = 1000,
        system: str = "",
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        trace: Any = None,
        **kwargs,
    ):
        """Tracked async wrapper around client.messages.create().

        Returns the full Anthropic response object (with .content[0].text,
        .usage, etc.) so callers can access all fields.
        """
        use_model = model or self.model_id

        input_summary = {
            "system": system[:200] if system else "",
            "messages": [{"role": m.get("role"), "content": m.get("content", "")[:200]} for m in messages],
        }
        generation = self._start_generation(
            name=f"anthropic-{self.agent_role}-create-message",
            input_data=input_summary,
            model=use_model,
            metadata={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "agent_role": self.agent_role,
            },
            trace=trace or self.trace,
        )

        try:
            start_time = datetime.now()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=use_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                ),
            )
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            output_text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }
            self._end_generation(generation, output_text, usage, duration_ms)
            return response
        except Exception as e:
            self._error_generation(generation, e)
            raise
