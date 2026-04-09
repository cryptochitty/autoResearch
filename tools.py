"""
Custom tools for the AutoResearch agents.
Each tool is a plain function decorated with @tool so CrewAI can bind it.
"""

import json
import io
import requests
import arxiv
from crewai.tools import tool
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# ArXiv search
# ---------------------------------------------------------------------------

@tool("arxiv_search")
def arxiv_search(query: str) -> str:
    """Search arXiv for academic papers. Input: a search query string.
    Returns JSON list of papers with title, authors, year, abstract, pdf_url."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for r in client.results(search):
            papers.append({
                "title": r.title,
                "authors": [a.name for a in r.authors[:4]],
                "year": r.published.year,
                "abstract": r.summary[:600],
                "pdf_url": r.pdf_url,
                "arxiv_id": r.entry_id,
            })
        return json.dumps(papers, indent=2)
    except Exception as exc:
        return f"arXiv search failed: {exc}"


# ---------------------------------------------------------------------------
# Web search (DuckDuckGo — no API key required)
# ---------------------------------------------------------------------------

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Input: a search query string.
    Returns JSON list of results with title, href, body."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        return json.dumps(results, indent=2)
    except Exception as exc:
        return f"Web search failed: {exc}"


# ---------------------------------------------------------------------------
# PDF reader (first 5 pages)
# ---------------------------------------------------------------------------

@tool("pdf_reader")
def pdf_reader(url: str) -> str:
    """Download and extract text from a PDF at the given URL (first 5 pages).
    Input: a direct PDF URL string."""
    try:
        import PyPDF2
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        reader = PyPDF2.PdfReader(io.BytesIO(response.content))
        pages = reader.pages[:5]
        text = "\n".join(p.extract_text() or "" for p in pages)
        return text[:4000]  # cap to avoid token overflow
    except Exception as exc:
        return f"PDF read failed: {exc}"
