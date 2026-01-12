"""
Tools for the portfolio agent.

These tools expose read-only portfolio data to the LLM so responses stay
strictly grounded in the provided source data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

DEFAULT_DATA_PATH = Path(__file__).with_name("portfolio_data.json")


class PortfolioTools:
    """Tools that return portfolio data from a local JSON file."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DEFAULT_DATA_PATH

    def _load_data(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            return {"error": f"Portfolio data file not found: {self.data_path}"}
        with self.data_path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)

    def _dump(self, data: Any) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _get_section(self, section: str) -> str:
        data = self._load_data()
        if "error" in data:
            return self._dump(data)
        return self._dump(data.get(section, []))

    def _sanitize_limit(self, limit: Optional[int], default: int = 10, max_limit: int = 50) -> int:
        try:
            if limit is None:
                return default
            limit_int = int(limit)
            if limit_int <= 0:
                return default
            return min(limit_int, max_limit)
        except (TypeError, ValueError):
            return default

    def get_profile(self) -> str:
        return self._get_section("profile")

    def get_contact(self) -> str:
        return self._get_section("contact")

    def get_links(self) -> str:
        return self._get_section("links")

    def get_about(self) -> str:
        data = self._load_data()
        if "error" in data:
            return self._dump(data)
        profile = data.get("profile", {})
        return self._dump({"about": profile.get("about")})

    def get_education(self) -> str:
        return self._get_section("education")

    def get_skills(self) -> str:
        return self._get_section("skills")

    def get_experience(self) -> str:
        return self._get_section("experience")

    def get_projects(self) -> str:
        return self._get_section("projects")

    def search_projects(self, keyword: str, limit: Optional[int] = 10) -> str:
        data = self._load_data()
        if "error" in data:
            return self._dump(data)
        if not keyword:
            return self._dump({"error": "Keyword is required."})

        keyword_lower = keyword.lower()
        projects = data.get("projects", [])
        matches = []
        for project in projects:
            haystack = " ".join([
                str(project.get("name", "")),
                str(project.get("summary", "")),
            ]).lower()
            if keyword_lower in haystack:
                matches.append(project)

        limit = self._sanitize_limit(limit)
        return self._dump(matches[:limit])


# JSON Tool definitions for LLM function calling (Ollama/OpenAI format)
def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get tool definitions in Ollama/OpenAI function calling format."""

    return [
        {
            "type": "function",
            "function": {
                "name": "get_profile",
                "description": "Get the core profile (name, tagline, about). Use for general introductions.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_contact",
                "description": "Get contact details (email). Use when the user asks how to reach out.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_links",
                "description": "Get public links (GitHub, LinkedIn, Instagram).",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_about",
                "description": "Get the about summary for the portfolio.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_education",
                "description": "Get education history.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_skills",
                "description": "Get the skills list grouped by category.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_experience",
                "description": "Get work and fellowship experience.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_projects",
                "description": "Get the portfolio projects.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_projects",
                "description": "Search projects by keyword in name or summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Search keyword (e.g., 'Power BI', 'Python')."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results. Default 10."
                        }
                    },
                    "required": ["keyword"]
                }
            }
        }
    ]


if __name__ == "__main__":
    tools = PortfolioTools()
    print(tools.get_profile())
