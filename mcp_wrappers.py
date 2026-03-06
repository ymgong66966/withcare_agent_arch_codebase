from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Literal

_mcp_logger = logging.getLogger(__name__)

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

from id_utils import new_uuid

TransportKind = Literal["http", "stdio"]

@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: TransportKind
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    keep_alive: bool = True

class MCPClientManager:
    def __init__(self) -> None:
        self._clients: Dict[str, Client] = {}

    def get_client(self, cfg: MCPServerConfig) -> Client:
        if cfg.name in self._clients:
            return self._clients[cfg.name]

        if cfg.transport == "http":
            transport = StreamableHttpTransport(url=cfg.url, headers=cfg.headers or {})
            client = Client(transport)
        else:
            transport = StdioTransport(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env if cfg.env is not None else dict(os.environ),
                cwd=cfg.cwd,
                keep_alive=cfg.keep_alive,
            )
            client = Client(transport)

        self._clients[cfg.name] = client
        return client

async def call_mcp_tool_patch(*, mgr: MCPClientManager, server: MCPServerConfig, tool_name: str, arguments: Dict[str, Any], purpose: str = "", timeout_s: float = 120, max_retries: int = 2) -> Dict[str, Any]:
    run_id = new_uuid("toolrun")

    tool_run = {"run_id": run_id, "tool": f"mcp:{server.name}:{tool_name}", "purpose": purpose, "args": arguments, "status": "ok"}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # Create a fresh client for each attempt to avoid stale connections
            client = mgr.get_client(server)
            async with client:
                result = await asyncio.wait_for(
                    client.call_tool(tool_name, arguments),
                    timeout=timeout_s,
                )

            # FastMCP CallToolResult stores the real payload in different places
            # depending on version.  Prefer structured_content (already a dict/list),
            # then fall back to parsing the text from the first content block.
            data = None
            sc = getattr(result, "structured_content", None)
            if sc is not None:
                # structured_content is typically {"result": <actual_data>}
                data = sc.get("result") if isinstance(sc, dict) else sc
            if data is None:
                # Fall back to parsing the text content
                content = getattr(result, "content", None) or []
                if content and hasattr(content[0], "text"):
                    import json as _mcp_json
                    try:
                        data = _mcp_json.loads(content[0].text)
                    except (ValueError, TypeError):
                        data = content[0].text

            return {"tools": {"tool_runs": [tool_run]}, "_mcp_result": {"run_id": run_id, "server": server.name, "tool": tool_name, "data": data}}
        except asyncio.TimeoutError:
            _mcp_logger.error(f"MCP tool {server.name}:{tool_name} timed out after {timeout_s}s (attempt {attempt}/{max_retries})")
            last_error = f"Timed out after {timeout_s}s"
            if attempt < max_retries:
                # Clear cached client to force fresh connection on retry
                mgr._clients.pop(server.name, None)
                await asyncio.sleep(1)
                continue
            break
        except Exception as e:
            _mcp_logger.error(f"MCP tool {server.name}:{tool_name} failed: {repr(e)} (attempt {attempt}/{max_retries})")
            last_error = repr(e)
            if attempt < max_retries:
                # Clear cached client to force fresh connection on retry
                mgr._clients.pop(server.name, None)
                await asyncio.sleep(2)
                continue
            break

    tool_run["status"] = "failed"
    tool_run["error"] = last_error
    return {"tools": {"tool_runs": [tool_run], "tool_failures": [{"tool": tool_run["tool"], "error": last_error, "recoverable": True}]}}


async def call_memory_tool(
    mgr: MCPClientManager,
    server: MCPServerConfig,
    tool_name: str,
    arguments: Dict[str, Any],
    fallback_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    timeout_ms: int = 5000,
) -> Dict[str, Any]:
    """
    Call a Memory MCP tool with timeout and fallback.

    Non-blocking: returns {} on timeout/failure if no fallback provided.
    If fallback_fn is given, calls it with the same arguments on failure.
    """
    try:
        result = await asyncio.wait_for(
            call_mcp_tool_patch(
                mgr=mgr, server=server,
                tool_name=tool_name, arguments=arguments,
            ),
            timeout=timeout_ms / 1000,
        )
        # Check for tool failures
        if result.get("tools", {}).get("tool_failures") and fallback_fn:
            _mcp_logger.warning(f"Memory MCP tool {tool_name} failed, using fallback")
            return await fallback_fn(**arguments)
        return result.get("_mcp_result", {}).get("data", {})
    except asyncio.TimeoutError:
        _mcp_logger.warning(f"Memory MCP tool {tool_name} timed out ({timeout_ms}ms)")
        if fallback_fn:
            return await fallback_fn(**arguments)
        return {}
    except Exception as e:
        _mcp_logger.warning(f"Memory MCP tool {tool_name} failed: {e}")
        if fallback_fn:
            return await fallback_fn(**arguments)
        return {}
