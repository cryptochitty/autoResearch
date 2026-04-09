"""
8 specialist agents, each with a focused role and only the tools it needs.
"""

from crewai import Agent, LLM
from tools import arxiv_search, web_search, pdf_reader
from config import ANTHROPIC_API_KEY, MODEL


def _llm(temperature: float = 0.5) -> LLM:
    return LLM(
        model=f"anthropic/{MODEL}",
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
    )


def create_agents() -> dict[str, Agent]:
    return {
        # ------------------------------------------------------------------
        "discovery": Agent(
            role="Research Discovery Specialist",
            goal="Find the 10–15 most relevant, recent, highly-cited papers on the topic.",
            backstory=(
                "You are an expert academic librarian with access to arXiv and the web. "
                "You systematically search multiple angles of a topic and return a clean, "
                "structured list of papers with metadata."
            ),
            tools=[arxiv_search, web_search],
            llm=_llm(0.3),
            verbose=True,
            max_iter=6,
        ),

        # ------------------------------------------------------------------
        "reader": Agent(
            role="Research Analyst",
            goal="Extract structured insights from each paper and synthesise cross-paper patterns.",
            backstory=(
                "PhD-level researcher skilled at skimming PDFs and identifying problem "
                "statements, methodologies, datasets, key results, and limitations. "
                "You produce rigorous per-paper breakdowns and comparison tables."
            ),
            tools=[pdf_reader, web_search],
            llm=_llm(0.4),
            verbose=True,
            max_iter=8,
        ),

        # ------------------------------------------------------------------
        "innovation": Agent(
            role="Research Innovation Strategist",
            goal="Identify research gaps and propose the single best novel research idea.",
            backstory=(
                "Creative researcher who spots unexplored intersections in existing work. "
                "You rigorously score ideas on feasibility, impact, and novelty before "
                "committing to the winner."
            ),
            tools=[web_search],
            llm=_llm(0.7),
            verbose=True,
            max_iter=4,
        ),

        # ------------------------------------------------------------------
        "validation": Agent(
            role="ML Research Feasibility Expert",
            goal="Turn the winning idea into a concrete, fully specified experiment plan.",
            backstory=(
                "Senior ML engineer who has shipped research to production. "
                "You find real public datasets, define evaluation metrics, estimate "
                "compute requirements, and break the work into implementable phases."
            ),
            tools=[web_search, arxiv_search],
            llm=_llm(0.4),
            verbose=True,
            max_iter=4,
        ),

        # ------------------------------------------------------------------
        "writer": Agent(
            role="Academic Research Paper Writer",
            goal="Write a complete, publication-quality IEEE-format paper (≥2 000 words).",
            backstory=(
                "Prolific academic author with 50+ IEEE publications. "
                "You write in a formal, clear tone, structure sections correctly, "
                "and only cite papers that were actually discovered in this session."
            ),
            tools=[],
            llm=_llm(0.5),
            verbose=True,
            max_iter=3,
        ),

        # ------------------------------------------------------------------
        "builder": Agent(
            role="ML Research Engineer",
            goal="Produce clean, runnable Python code that implements the proposed research.",
            backstory=(
                "Senior ML engineer who translates research designs into self-contained "
                "Python scripts. You use standard libraries, add clear comments, and "
                "always specify where to get the data."
            ),
            tools=[web_search],
            llm=_llm(0.3),
            verbose=True,
            max_iter=4,
        ),

        # ------------------------------------------------------------------
        "evaluation": Agent(
            role="Research Benchmarking Specialist",
            goal="Produce a rigorous comparison of the proposed method vs existing baselines.",
            backstory=(
                "Expert in ML evaluation who designs ablation studies and builds "
                "honest comparison tables. You report trade-offs, not just wins."
            ),
            tools=[web_search],
            llm=_llm(0.4),
            verbose=True,
            max_iter=3,
        ),

        # ------------------------------------------------------------------
        "explainer": Agent(
            role="Research Communication Expert",
            goal="Convert the full research output into a simple explanation and a compelling pitch.",
            backstory=(
                "Science communicator who makes complex research accessible. "
                "You write for both technical and non-technical audiences and "
                "craft punchy hackathon pitches."
            ),
            tools=[],
            llm=_llm(0.6),
            verbose=True,
            max_iter=2,
        ),
    }
