"""
Risk Extraction Prompt Templates & JSON Schema

Defines the system prompt, user prompt templates, and the strict JSON schema
used by the LLM to produce structured risk profiles.
"""

from __future__ import annotations

# ============================================================
# JSON Output Schema
# ============================================================

RISK_PROFILE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": "Company ticker symbol"},
        "risk_category": {"type": "string", "description": "Risk category name"},
        "is_present": {"type": "boolean", "description": "Whether this risk is mentioned/present"},
        "severity": {
            "type": "string",
            "enum": ["negligible", "low", "medium", "high", "critical"],
            "description": "Estimated severity level on 5-point scale",
        },
        "explanation": {
            "type": "string",
            "description": "1-3 sentence explanation of the risk based on evidence",
        },
        "evidence_snippets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Direct quotes from the source text supporting the assessment",
            "minItems": 1,
            "maxItems": 3,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence level in the assessment (0.0 to 1.0)",
        },
    },
    "required": [
        "company",
        "risk_category",
        "is_present",
        "severity",
        "explanation",
        "evidence_snippets",
        "confidence",
    ],
}

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are a senior financial risk analyst specializing in SEC 10-K filings. You produce precise, differentiated risk assessments. You NEVER default to generic scores.

STRICT RULES:
1. Base your assessment ONLY on the provided evidence text. Do NOT use external knowledge.
2. Output ONLY valid JSON matching the specified schema. No extra text before or after the JSON.
3. Every claim in your explanation MUST be directly supported by the evidence snippets.
4. If the evidence does not clearly mention the risk category, set is_present to false and severity to "negligible".
5. Evidence snippets must be DIRECT QUOTES from the provided text — not paraphrased.
6. Keep explanations concise: 1-3 sentences maximum.

CRITICAL — CONFIDENCE SCORING:
You MUST vary your confidence scores meaningfully. Do NOT default to 0.94 or round numbers.
Think step-by-step about how strong the evidence actually is:
- 0.90-0.99: Multiple explicit, detailed paragraphs directly discussing this specific risk with quantified impact
- 0.75-0.89: Clear discussion of the risk but lacking specific financial figures or detailed mitigation plans
- 0.55-0.74: Risk is mentioned but only in passing, within broader context, or as boilerplate language
- 0.30-0.54: Only tangentially related evidence; requires significant interpretation
- 0.05-0.29: No meaningful evidence for this risk category

CRITICAL — SEVERITY DIFFERENTIATION:
SEC 10-K risk factors are written defensively: almost EVERY risk factor claims it "could have a material adverse effect on our business, financial condition and results of operations." Because this phrase is universal boilerplate, it is NOT by itself evidence of high severity — ignore it when scoring.

Default to "medium" for ordinary, well-discussed risks. Escalate to "high" or "critical" ONLY when the evidence contains at least one concrete ESCALATOR beyond boilerplate:
- a quantified financial impact (a specific dollar or percentage figure), OR
- a named, active legal/regulatory action (a specific lawsuit, agency, or settlement), OR
- a past incident that ACTUALLY occurred (not a hypothetical "could"), OR
- explicit existential / enterprise-threatening language.
If NONE of these escalators are present, the severity ceiling is "medium" — do not exceed it no matter how serious the boilerplate sounds."""

# ============================================================
# User Prompt Template
# ============================================================

USER_PROMPT_TEMPLATE = """Analyze the following evidence chunks from {company_name} ({ticker})'s 10-K filing for the risk category: **{risk_category}**.

Risk Category Description: {risk_description}

--- EVIDENCE CHUNKS ---
{evidence_text}
--- END EVIDENCE ---

Based ONLY on the evidence above, produce a JSON risk assessment with this exact schema:
{{
    "company": "{ticker}",
    "risk_category": "{risk_category}",
    "is_present": true/false,
    "severity": "negligible" | "low" | "medium" | "high" | "critical",
    "explanation": "1-3 sentence explanation based on evidence",
    "evidence_snippets": ["direct quote 1", "direct quote 2"],
    "confidence": 0.0 to 1.0
}}

SEVERITY SCALE (an ESCALATOR = a quantified figure, a named active lawsuit/regulator, an incident that actually occurred, or existential language):
- "critical": An escalator describing an existential / enterprise-threatening situation — an ongoing crisis, a quantified massive loss, or a regulatory action that could shut down a core business line.
- "high": Contains a clear escalator — e.g. specific dollar/percentage impact, a named active legal/regulatory proceeding, or a past incident that already occurred.
- "medium": A risk that is MEANINGFULLY DISCUSSED — the filing describes how it arises, gives concrete examples, or outlines mitigation — but with no escalator. This is the typical, well-managed risk factor.
- "low": The category is only mentioned in passing — named or listed among many risks with little to no specific discussion, mechanism, or detail. If the evidence is thin/sparse and merely references the topic, prefer "low" over "medium".
- "negligible": Not meaningfully discussed; only tangential references or no relevant evidence.

DISTINCTION — low vs medium: "medium" requires actual discussion/detail; "low" is a bare mention. Do not inflate a brief, generic reference to "medium".

REMEMBER: "could have a material adverse effect" is boilerplate present in nearly every risk factor — it NEVER justifies "high" on its own. Without a concrete escalator, the answer is "medium" at most.

Output ONLY the JSON object, nothing else:"""


# ============================================================
# Few-shot Examples (anchor the medium-vs-high distinction)
# ============================================================
# Generic, non-company-specific examples so the model learns the SCORING RULE
# rather than memorizing any filing. Kept as a plain string (no .format) so the
# JSON braces below do not interfere with template substitution.

FEW_SHOT_EXAMPLES = """Scoring examples (note: boilerplate "could ... material adverse effect" stays "medium"; a concrete escalator earns "high"):

EX1 (boilerplate, no escalator) Evidence: "We depend on a limited number of suppliers; if they limit supply or raise prices it could have a material adverse effect on our business." -> {"severity": "medium", "is_present": true} (hypothetical, no figures/named party/incident)

EX2 (escalator present) Evidence: "In fiscal 2024 we recorded a $450M charge from an ongoing EU antitrust investigation; we have paid $120M in fines and face up to $1.2B more." -> {"severity": "high", "is_present": true} (quantified impact + active proceeding)

EX3 (bare mention) Evidence (for Macroeconomic Risk): "Among other factors, our results may be affected by changes in tax laws, interest rates, and general economic conditions." -> {"severity": "low", "is_present": true} (listed in passing among many factors, no real discussion)

EX4 (off-topic) Evidence (for Cybersecurity Risk): "Our equipment requires periodic maintenance." -> {"severity": "negligible", "is_present": false} (no relevant evidence)

Now complete the REAL task below using the same rules."""


def format_evidence_chunks(chunks: list[dict], char_limit: int = None) -> str:
    """
    Format retrieved evidence chunks into a string for the prompt.

    Args:
        chunks: List of chunk dicts with 'text', 'company', 'chunk_id' fields
        char_limit: If set, truncate each chunk's text to this many characters
            (keeps requests within the LLM token budget).

    Returns:
        Formatted evidence string
    """
    evidence_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]
        if char_limit and len(text) > char_limit:
            text = text[:char_limit].rstrip() + " …"
        evidence_parts.append(
            f"[Chunk {i}] (ID: {chunk.get('chunk_id', 'unknown')})\n{text}"
        )
    return "\n\n".join(evidence_parts)


def build_prompt(
    ticker: str,
    company_name: str,
    risk_category: str,
    risk_description: str,
    evidence_chunks: list[dict],
    evidence_char_limit: int = None,
) -> tuple[str, str]:
    """
    Build the complete system + user prompt for risk extraction.

    Args:
        ticker: Company ticker symbol
        company_name: Full company name
        risk_category: Name of the risk category
        risk_description: Description of the risk category
        evidence_chunks: List of retrieved evidence chunk dicts
        evidence_char_limit: Optional per-chunk character cap (token budget).

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    evidence_text = format_evidence_chunks(evidence_chunks, char_limit=evidence_char_limit)

    task_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        risk_category=risk_category,
        risk_description=risk_description,
        evidence_text=evidence_text,
    )

    # Prepend few-shot examples to anchor the severity scoring rule.
    user_prompt = f"{FEW_SHOT_EXAMPLES}\n\n{task_prompt}"

    return SYSTEM_PROMPT, user_prompt
