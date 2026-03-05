from __future__ import annotations
from fastmcp import FastMCP

mcp = FastMCP("DemoWithCareServer")

@mcp.tool
def nearby_providers(query: str, location: str = "UNKNOWN") -> dict:
    # Demo "search-like" tool. In production, replace with Firecrawl/GoogleMaps/etc.
    return {
        "location": location,
        "query": query,
        "results": [
            {"title": "Demo Result A", "contact": "555-0101", "notes": "placeholder"},
            {"title": "Demo Result B", "contact": "555-0122", "notes": "placeholder"},
        ],
        "disclaimer": "Demo data only.",
    }

@mcp.tool
def medicaid_checklist(state: str = "IL") -> dict:
    # Demo "guidance-like" tool. In production, domain expert agent usually doesn't need a tool here.
    return {"state": state, "steps": ["Collect docs", "Check eligibility", "Apply/verify"], "disclaimer": "Demo only."}

if __name__ == "__main__":
    mcp.run()
