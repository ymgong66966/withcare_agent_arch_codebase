"""
Real search MCP server for WithCare deep search.

Tools:
  - google_places_search: Google Places API text search
  - general_online_search_with_one_query: Firecrawl web search
  - website_map: Firecrawl /v2/map endpoint for finding URLs
  - scrape_multiple_websites_after_website_map: Firecrawl structured JSON extraction

Env vars required: GOOGLE_MAPS_API_KEY, FIRECRAWL_API_KEY
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import List
from urllib.parse import urlparse

import sys
import aiohttp
import httpx
from fastmcp import FastMCP

mcp = FastMCP("WithCareSearchServer")


def _log(msg: str) -> None:
    """Print to stderr so logs show in the parent process terminal."""
    print(f"[search_mcp] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# google_places_search
# ---------------------------------------------------------------------------

@mcp.tool
async def google_places_search(location: str, location_query: str = "") -> list[dict]:
    """Search for places using Google Places API text search.

    Args:
        location: Location string, e.g. "Chicago, IL"
        location_query: Optional service type, e.g. "in-home care agency"

    Returns a list of dicts with name, address, rating, website, domain.
    """
    query = location_query
    _log(f">>> google_places_search called | location={location!r} location_query={query!r}")
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        return [{"error": "GOOGLE_MAPS_API_KEY not set"}]

    text_query = f"{query} in {location}" if query else location
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.rating,places.websiteUri,places.location"
        ),
    }
    payload = {"textQuery": text_query, "maxResultCount": 5}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as e:
        return [{"error": f"Google Places API request failed: {e}"}]

    places = data.get("places", [])
    _log(f"<<< google_places_search returned {len(places)} places")
    results = []
    for p in places:
        display = p.get("displayName") or {}
        website = p.get("websiteUri") or ""
        domain = _extract_domain(website) if website else None
        results.append({
            "name": display.get("text", ""),
            "address": p.get("formattedAddress", ""),
            "rating": p.get("rating"),
            "website": website,
            "domain": domain,
        })
    return results


def _extract_domain(url: str) -> str | None:
    """Extract a useful root URL from a Places websiteUri.

    Keeps geo-location path segments (e.g.
    homeinstead.com/home-care/usa/ca/san-francisco) instead of stripping
    to bare domain, but removes query params and overly deep paths.
    Uses urlparse — no OpenAI dependency.
    """
    try:
        parsed = urlparse(url)
        # Keep scheme + netloc + path up to 5 segments
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        # Heuristic: keep path segments that look geographic or categorical
        # but cap at 5 to avoid overly specific pages
        trimmed = "/".join(parts[:5]) if parts else ""
        root = f"{parsed.scheme}://{parsed.netloc}"
        return f"{root}/{trimmed}" if trimmed else root
    except Exception:
        return url


# ---------------------------------------------------------------------------
# general_online_search
# ---------------------------------------------------------------------------

@mcp.tool
async def general_online_search_with_one_query(query: str) -> list[dict]:
    """Quick web search via Firecrawl. Best for general knowledge queries.

    Args:
        query: Search query string

    Returns a list of search result dicts.
    """
    _log(f">>> general_online_search_with_one_query called | query={query!r}")
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return [{"error": "FIRECRAWL_API_KEY not set"}]

    try:
        from firecrawl import Firecrawl

        app = Firecrawl(api_key=api_key)
        search_result = app.search(sources=["web"], query=query, limit=3)
        _log(f"<<< general_online_search_with_one_query got result type={type(search_result).__name__}")

        if not search_result:
            return [{"result": f"No results found for: {query}"}]

        if isinstance(search_result, list):
            return search_result or [{"result": f"No results found for: {query}"}]
        if isinstance(search_result, dict):
            for key in ("data", "results"):
                if key in search_result:
                    return search_result[key] or [{"result": f"No results found for: {query}"}]
            return [search_result]
        return [{"result": str(search_result)}]

    except Exception as e:
        _log(f"!!! general_online_search_with_one_query error: {e}")
        return [{"error": f"Search failed: {e}"}]


# ---------------------------------------------------------------------------
# website_map
# ---------------------------------------------------------------------------

@mcp.tool
async def website_map(url: str, search_queries: list[str]) -> list[dict]:
    """Find relevant URLs on a web domain for given queries via Firecrawl /v2/map.

    Args:
        url: Root domain URL, e.g. "https://example.com"
        search_queries: Up to 3 short query strings

    Returns a list of dicts with url, title, description.
    Rate-limited: 2s between API calls.
    """
    _log(f">>> website_map called | url={url!r} search_queries={search_queries!r}")
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return [{"error": "FIRECRAWL_API_KEY not set"}]

    search_queries = search_queries[:3]  # enforce max 3
    endpoint = "https://api.firecrawl.dev/v2/map"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    result_urls: set[str] = set()
    result_ls: list[dict] = []

    for i, search in enumerate(search_queries):
        if i > 0:
            await asyncio.sleep(2.0)

        payload = {"url": url, "search": search, "limit": 3, "sitemap": "include"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    for link in data.get("links", []):
                        link_url = link if isinstance(link, str) else link.get("url", "")
                        if link_url and link_url not in result_urls:
                            result_urls.add(link_url)
                            if isinstance(link, dict):
                                result_ls.append(link)
                            else:
                                result_ls.append({"url": link_url})
                elif response.status_code == 429:
                    continue
                else:
                    continue
        except Exception:
            continue

    _log(f"<<< website_map returned {len(result_ls)} URLs")
    return result_ls or [{"info": f"No URLs found for {url}"}]


# ---------------------------------------------------------------------------
# scrape_websites
# ---------------------------------------------------------------------------

@mcp.tool
async def scrape_multiple_websites_after_website_map(urls: list[str], queries: list[str]) -> list[dict]:
    """Scrape up to 5 URLs and extract structured answers to queries via Firecrawl.

    Args:
        urls: List of URLs to scrape (max 5)
        queries: List of query strings to extract answers for

    Returns a list of dicts with url, status, content.
    """
    _log(f">>> scrape_multiple_websites_after_website_map called | urls={urls!r} queries={queries!r}")
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return [{"error": "FIRECRAWL_API_KEY not set"}]
    if not urls:
        return [{"error": "No URLs provided"}]
    if not queries:
        return [{"error": "No queries provided"}]

    urls = urls[:5]

    try:
        from firecrawl import Firecrawl
        from pydantic import BaseModel

        class QueryResponse(BaseModel):
            queries: str
            answer: str

        app = Firecrawl(api_key=api_key)
        extract_prompt = (
            f"queries: {str(queries)}\n"
            "answer: answer with the information you get from the website. "
            "if cannot find answer, return 'answer not found'"
        )

        async def scrape_single(u: str) -> dict:
            try:
                result = app.scrape(
                    u,
                    formats=[{
                        "type": "json",
                        "schema": QueryResponse,
                        "prompt": extract_prompt,
                    }],
                    only_main_content=True,
                    timeout=30000,
                )
                if result and hasattr(result, "json") and result.json:
                    return {"url": u, "status": "success", "content": result.json}
                return {"url": u, "status": "error", "error": "No content extracted"}
            except Exception as e:
                return {"url": u, "status": "error", "error": str(e)}

        results = await asyncio.gather(*(scrape_single(u) for u in urls), return_exceptions=True)
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({"url": urls[i], "status": "error", "error": str(r)})
            elif r.get("status") == "success":
                answer = (r.get("content") or {}).get("answer", "")
                if "answer not found" not in answer.lower():
                    processed.append(r)
            else:
                processed.append(r)
        _log(f"<<< scrape_multiple_websites_after_website_map returned {len(processed)} results")
        return processed or [{"info": "No useful content extracted from any URL"}]

    except Exception as e:
        _log(f"!!! scrape_multiple_websites_after_website_map error: {e}")
        return [{"error": f"Scraping setup failed: {e}"}]


if __name__ == "__main__":
    mcp.run()
