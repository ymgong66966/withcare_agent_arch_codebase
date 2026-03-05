"""
Anthropic Claude client with Langfuse tracking support.
"""

from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Please install anthropic: pip install anthropic")

try:
    from langfuse import Langfuse
    langfuse = Langfuse()
except ImportError:
    logging.warning("Langfuse not available. Install with: pip install langfuse")
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

    def converse(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """
        Synchronous conversation with Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters for the API

        Returns:
            Response text from Claude
        """
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            **kwargs
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""


class TrackedAnthropicClient(AnthropicClient):
    """AnthropicClient with Langfuse tracking"""

    def __init__(
        self,
        session_id: str = None,
        agent_role: str = None,
        user_id: str = None,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        super().__init__(model_id=model_id, api_key=api_key)
        self.session_id = session_id or f"anthropic-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.agent_role = agent_role or "unknown-agent"
        self.user_id = user_id or "anonymous"

    def simple_chat(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Simple chat with Langfuse logging"""
        # Create a generation trace with error handling
        generation = None
        if langfuse:
            try:
                generation = langfuse.generation(
                    name=f"anthropic-{self.agent_role}-simple-chat",
                    model=self.model_id,
                    input=prompt,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    metadata={
                        "max_tokens": max_tokens,
                        "system": "langgraph-kafka",
                        "component": "task-generator",
                        **kwargs
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse generation: {e}")
                generation = None

        try:
            # Call the parent converse method directly
            start_time = datetime.now()
            messages = [{"role": "user", "content": prompt}]
            response = super(TrackedAnthropicClient, self).converse(messages, max_tokens=max_tokens)
            end_time = datetime.now()

            # Calculate approximate token counts (rough estimation)
            input_tokens = len(prompt.split()) * 1.3  # Rough approximation
            output_tokens = len(response.split()) * 1.3

            # Update the generation with response
            if generation:
                try:
                    generation.end(
                        output=response,
                        usage={
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                            "total": int(input_tokens + output_tokens)
                        },
                        metadata={
                            "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                            "model": self.model_id
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"Failed to end Langfuse generation: {langfuse_error}")

            return response

        except Exception as e:
            # Log the error
            if generation:
                try:
                    generation.end(
                        level="ERROR",
                        metadata={"error": str(e)}
                    )
                except Exception as langfuse_error:
                    logger.warning(f"Failed to log error to Langfuse: {langfuse_error}")
            raise

    async def async_chat(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """Async chat with Langfuse logging"""
        # Create a generation trace with error handling
        generation = None
        if langfuse:
            try:
                generation = langfuse.generation(
                    name=f"anthropic-{self.agent_role}-async-chat",
                    model=self.model_id,
                    input=prompt,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    metadata={
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": "langgraph-kafka",
                        "component": "task-generator",
                        **kwargs
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create Langfuse generation: {e}")
                generation = None

        try:
            # Call the parent converse method in async context
            start_time = datetime.now()
            messages = [{"role": "user", "content": prompt}]
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: super(TrackedAnthropicClient, self).converse(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            end_time = datetime.now()

            # Calculate approximate token counts
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response.split()) * 1.3

            # Update the generation with response
            if generation:
                try:
                    generation.end(
                        output=response,
                        usage={
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                            "total": int(input_tokens + output_tokens)
                        },
                        metadata={
                            "duration_ms": int((end_time - start_time).total_seconds() * 1000),
                            "temperature": temperature,
                            "model": self.model_id
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"Failed to end Langfuse generation: {langfuse_error}")

            return response

        except Exception as e:
            # Log the error
            if generation:
                try:
                    generation.end(
                        level="ERROR",
                        metadata={"error": str(e)}
                    )
                except Exception as langfuse_error:
                    logger.warning(f"Failed to log error to Langfuse: {langfuse_error}")
            raise
