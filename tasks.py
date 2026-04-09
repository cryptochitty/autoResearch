"""
8 tasks wired in sequence via `context` so each agent sees prior outputs.
"""

from textwrap import dedent
from crewai import Task


def create_tasks(agents: dict, topic: str) -> list[Task]:

    # 1 ── Discovery ──────────────────────────────────────────────────────
    t_discovery = Task(
        description=dedent(f"""
            Search for the 10–15 most relevant academic papers on:
            "{topic}"

            Use arxiv_search with at least 2–3 different query phrasings.
            Use web_search to find any notable papers not on arXiv
            (IEEE, Springer, Nature, etc.).

            For each paper return:
            - title, authors (up to 4), year
            - abstract (≤ 400 words)
            - key technique / method keyword
            - pdf_url (if available)
            - source (arXiv / IEEE / other)

            Output: JSON array named "papers".
        """),
        expected_output=(
            "JSON array of 10–15 paper objects, each with title, authors, year, "
            "abstract, technique, pdf_url, source."
        ),
        agent=agents["discovery"],
    )

    # 2 ── Reader ─────────────────────────────────────────────────────────
    t_reader = Task(
        description=dedent(f"""
            Analyse ALL papers discovered for "{topic}".

            For each paper:
            1. Problem statement (1–2 sentences)
            2. Methodology & key techniques
            3. Dataset used
            4. Best reported metric / result
            5. Limitations explicitly stated or implied

            If a pdf_url is available, use pdf_reader to get deeper detail.

            Conclude with:
            - A 3-column comparison table (Paper | Approach | Main Result)
            - Patterns that appear in ≥ 3 papers
            - The most common weaknesses across the literature
        """),
        expected_output=(
            "Per-paper analysis + comparison table + cross-paper patterns + "
            "shared weaknesses."
        ),
        agent=agents["reader"],
        context=[t_discovery],
    )

    # 3 ── Innovation ─────────────────────────────────────────────────────
    t_innovation = Task(
        description=dedent(f"""
            Using the literature analysis for "{topic}":

            Step 1 — List 4–6 concrete research gaps.
            Step 2 — Generate 3 novel research ideas that fill one or more gaps.
            Step 3 — Score each idea (1–10) on:
                      • Feasibility  (can a team of 2 implement in ~3 months?)
                      • Impact       (meaningful improvement over SOTA?)
                      • Novelty      (not done before?)
            Step 4 — Select the WINNER. Explain:
                      • What gap it fills
                      • How it differs from existing work
                      • Expected contribution (metric improvement, new capability, etc.)
        """),
        expected_output=(
            "Gaps list + 3 ideas with scores + winning idea with clear justification."
        ),
        agent=agents["innovation"],
        context=[t_reader],
    )

    # 4 ── Validation ─────────────────────────────────────────────────────
    t_validation = Task(
        description=dedent(f"""
            Design a full experiment plan for the winning idea on "{topic}":

            1. Formal problem formulation (use math notation where helpful)
            2. Proposed model / algorithm (architecture or pseudocode)
            3. Primary dataset — prefer free public sources (Kaggle, UCI, HuggingFace,
               government open data). Include download URL.
            4. Evaluation metrics with justification
            5. Baselines to compare against (≥ 3)
            6. Estimated compute (GPU hours, RAM)
            7. 4-week implementation roadmap (week-by-week milestones)
            8. Feasibility verdict: GO / CONDITIONAL GO / NO GO + reason
        """),
        expected_output=(
            "Complete experiment design: problem formulation, architecture, dataset, "
            "metrics, baselines, compute estimate, roadmap, feasibility verdict."
        ),
        agent=agents["validation"],
        context=[t_innovation],
    )

    # 5 ── Writer ─────────────────────────────────────────────────────────
    t_writer = Task(
        description=dedent(f"""
            Write a full IEEE-format research paper for "{topic}".

            Required sections (do not skip any):
            [TITLE]       — concise, descriptive
            [ABSTRACT]    — 150–250 words
            [KEYWORDS]    — 5–7 terms
            [I. INTRODUCTION]     — motivation, problem, contributions, paper outline
            [II. RELATED WORK]    — grouped by theme; cite only papers from discovery
            [III. METHODOLOGY]    — detailed; include equations if relevant
            [IV. SYSTEM ARCHITECTURE] — describe components and data flow
            [V. EXPERIMENTAL SETUP]   — datasets, metrics, baselines, hardware
            [VI. RESULTS AND DISCUSSION] — tables, analysis, comparison
            [VII. CONCLUSION]     — summary of contributions
            [VIII. FUTURE WORK]   — 3–4 directions
            [REFERENCES]          — IEEE format: [N] A. Author, "Title," Venue, Year.

            Rules:
            - Minimum 2 000 words (aim for 3 000+)
            - Academic tone throughout
            - Only cite papers that appear in the discovery list
            - No placeholder text — every section must be substantive
        """),
        expected_output="Complete IEEE-format research paper, ≥ 2 000 words, all sections filled.",
        agent=agents["writer"],
        context=[t_discovery, t_reader, t_innovation, t_validation],
    )

    # 6 ── Builder ────────────────────────────────────────────────────────
    t_builder = Task(
        description=dedent(f"""
            Write complete, runnable Python code for "{topic}" that implements
            the experiment design from the validation phase.

            Include these files (each clearly delimited with # === filename ===):

            # === requirements.txt ===
            (all pip packages needed)

            # === data_loader.py ===
            (download / load / split the dataset)

            # === model.py ===
            (model / algorithm definition)

            # === train.py ===
            (training loop with epoch logging)

            # === evaluate.py ===
            (compute metrics, print results table, save plots)

            # === main.py ===
            (ties everything together; runnable with `python main.py`)

            Rules:
            - Use only standard ML libs: scikit-learn, PyTorch, TensorFlow, pandas,
              matplotlib, seaborn, numpy
            - Add comments on non-obvious logic
            - State where to get the dataset at the top of data_loader.py
            - Code must run without modification on a machine that has the packages
        """),
        expected_output=(
            "Complete Python codebase split into 6 files: requirements.txt, "
            "data_loader.py, model.py, train.py, evaluate.py, main.py."
        ),
        agent=agents["builder"],
        context=[t_validation, t_innovation],
    )

    # 7 ── Evaluation ─────────────────────────────────────────────────────
    t_evaluation = Task(
        description=dedent(f"""
            Produce a benchmarking report for "{topic}":

            1. Performance comparison table
               Columns: Method | Dataset | Primary Metric | Score | Notes
               Rows: proposed method + ≥ 3 baselines from literature
               Use realistic expected numbers; flag as "projected" if not measured.

            2. Advantages of proposed approach (bullet list)

            3. Limitations and trade-offs (honest; at least 3)

            4. Ablation study plan
               - List 3–4 components to ablate
               - State what each ablation tests

            5. Reproducibility checklist
               □ Dataset publicly available
               □ Code released
               □ Hyperparameters reported
               □ Random seeds fixed
        """),
        expected_output=(
            "Comparison table + advantages + trade-offs + ablation plan + "
            "reproducibility checklist."
        ),
        agent=agents["evaluation"],
        context=[t_validation, t_builder, t_reader],
    )

    # 8 ── Explainer ──────────────────────────────────────────────────────
    t_explainer = Task(
        description=dedent(f"""
            Create three human-friendly outputs for "{topic}":

            ── SECTION A: Simple Explanation (ELI5, ≤ 200 words) ──
            What problem? How does it work (zero jargon)? Why does it matter?

            ── SECTION B: 60-Second Hackathon Pitch ──
            Hook (1 sentence) | Problem (2 sentences) | Solution (3 sentences) |
            Demo description | Impact (1 sentence)

            ── SECTION C: Key Innovation Points ──
            3 bullet points — what is genuinely new, the technical contribution,
            and the real-world impact.

            ── SECTION D: Next Steps (action items) ──
            3 concrete things a team should do in the next 2 weeks to move this forward.
        """),
        expected_output=(
            "4 sections: ELI5 explanation, pitch, key innovation bullets, next steps."
        ),
        agent=agents["explainer"],
        context=[t_innovation, t_evaluation, t_writer],
    )

    return [
        t_discovery,
        t_reader,
        t_innovation,
        t_validation,
        t_writer,
        t_builder,
        t_evaluation,
        t_explainer,
    ]
