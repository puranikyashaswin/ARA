"""
Search tool using Tavily API.
"""
from typing import Annotated
from langchain_core.tools import tool
from tavily import TavilyClient
import os


@tool
def web_search(
    query: Annotated[str, "The search query to find information on the web."]
) -> str:
    """
    Search the web for current information using Tavily.
    Use this when you need up-to-date information, facts, or need to verify something.
    Returns relevant search results with sources.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        
        results = []
        
        # Include Tavily's direct answer if available
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}\n")
        
        # Include individual results
        results.append("Sources:")
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")[:500]  # Truncate for context
            results.append(f"\n{i}. {title}\n   URL: {url}\n   {content}")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Search failed: {str(e)}"
