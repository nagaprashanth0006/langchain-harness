"""Query router — dedicated classifier pass."""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List

from .config import get_classifier_backend, get_eligible_sources
from .llm_client import LLMClient
from .models import QueryCategory, RouterOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a query classifier for a research assistant. Analyze the user query and respond with a JSON object.

Categories:
- empirical_factual: news events, scientific facts, medical facts, statistics, historical events
- procedural_reference: legal procedures, how-to guides, technical documentation, regulations
- interpretive_subjective: relationships, opinions, "should I...", ethics, personal advice framing
- personal_sensitive: mental health, financial advice, legal advice for individuals, medical advice for individuals

Judge rubric options:
- corroboration: for empirical/procedural — independent source agreement matters
- reasoning_quality: for interpretive — logical coherence and perspective balance matters
- sensitivity_balanced: for personal_sensitive — hedging, professional referral, balanced perspectives

Return ONLY valid JSON with these fields:
{
  "category": "<primary category>",
  "secondary_category": "<secondary or null>",
  "eligible_sources": ["list", "of", "source", "ids"],
  "judge_rubric": "<rubric>",
  "sensitivity": <true/false>,
  "reasoning": "<brief explanation>"
}

Available source IDs: ddg, pubmed, semantic_scholar, openalexs, crossref, arxiv, unpaywall"""

_EXAMPLES = [
    {
        "query": "What are the side effects of metformin?",
        "result": {
            "category": "empirical_factual",
            "secondary_category": "personal_sensitive",
            "eligible_sources": ["ddg", "pubmed", "semantic_scholar"],
            "judge_rubric": "corroboration",
            "sensitivity": True,
            "reasoning": "Medical drug information — needs corroboration from medical sources; personal health implications require sensitivity flag",
        },
    },
    {
        "query": "How should I handle conflict with my partner?",
        "result": {
            "category": "interpretive_subjective",
            "secondary_category": "personal_sensitive",
            "eligible_sources": ["ddg"],
            "judge_rubric": "reasoning_quality",
            "sensitivity": True,
            "reasoning": "Relationship advice — interpretive and personal; no academic APIs appropriate",
        },
    },
    {
        "query": "Latest research on CRISPR gene editing",
        "result": {
            "category": "empirical_factual",
            "secondary_category": None,
            "eligible_sources": ["ddg", "pubmed", "semantic_scholar", "openalexs", "crossref", "arxiv"],
            "judge_rubric": "corroboration",
            "sensitivity": False,
            "reasoning": "Scientific research query — all academic sources appropriate; factual corroboration matters",
        },
    },
]


def _build_messages(query: str) -> List[Dict[str, str]]:
    examples_text = "\n".join(
        f"Query: {ex['query']}\nResponse: {json.dumps(ex['result'])}" for ex in _EXAMPLES
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Examples:\n{examples_text}\n\nNow classify this query:\n{query}",
        },
    ]


def _parse_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise


def _fallback_classify(query: str) -> Dict[str, Any]:
    q_lower = query.lower()
    medical_terms = {"drug", "medication", "symptom", "disease", "diagnosis", "treatment", "dose", "dosage", "medicine", "therapy", "side effect", "clinical"}
    legal_terms = {"law", "legal", "regulation", "statute", "court", "contract"}
    academic_terms = {"research", "study", "paper", "journal", "crispr", "molecule", "gene", "protein"}
    personal_terms = {"i ", "my ", "should i", "am i", "will i", "can i", "help me"}

    has_medical = any(t in q_lower for t in medical_terms)
    has_legal = any(t in q_lower for t in legal_terms)
    has_academic = any(t in q_lower for t in academic_terms)
    has_personal = any(t in q_lower for t in personal_terms)

    if has_personal and (has_medical or has_legal):
        return {
            "category": "personal_sensitive",
            "secondary_category": "empirical_factual" if has_medical else "procedural_reference",
            "eligible_sources": ["ddg"],
            "judge_rubric": "sensitivity_balanced",
            "sensitivity": True,
            "reasoning": "Fallback classifier: personal + domain-specific query",
        }
    elif has_medical or has_academic:
        sources = ["ddg", "pubmed", "semantic_scholar", "openalexs", "arxiv"]
        return {
            "category": "empirical_factual",
            "secondary_category": None,
            "eligible_sources": sources,
            "judge_rubric": "corroboration",
            "sensitivity": has_medical,
            "reasoning": "Fallback classifier: academic/medical query",
        }
    elif has_legal:
        return {
            "category": "procedural_reference",
            "secondary_category": None,
            "eligible_sources": ["ddg", "crossref"],
            "judge_rubric": "corroboration",
            "sensitivity": False,
            "reasoning": "Fallback classifier: legal/procedural query",
        }
    else:
        return {
            "category": "empirical_factual",
            "secondary_category": None,
            "eligible_sources": ["ddg"],
            "judge_rubric": "corroboration",
            "sensitivity": False,
            "reasoning": "Fallback classifier: general query",
        }


async def route_query(query: str) -> RouterOutput:
    backend = get_classifier_backend()
    client = LLMClient(backend)
    messages = _build_messages(query)

    try:
        raw = await client.chat(messages, temperature=0.0, max_tokens=512, json_mode=True)
        data = _parse_response(raw)
    except Exception as e:
        logger.warning(f"Router LLM failed ({e}), using fallback classifier")
        data = _fallback_classify(query)

    try:
        category = QueryCategory(data.get("category", "empirical_factual"))
    except ValueError:
        category = QueryCategory.EMPIRICAL_FACTUAL

    secondary_raw = data.get("secondary_category")
    secondary = None
    if secondary_raw:
        try:
            secondary = QueryCategory(secondary_raw)
        except ValueError:
            pass

    eligible = data.get("eligible_sources") or get_eligible_sources(category.value)
    rubric = data.get("judge_rubric") or "corroboration"
    sensitivity = bool(data.get("sensitivity", False))

    return RouterOutput(
        category=category,
        secondary_category=secondary,
        eligible_sources=eligible,
        judge_rubric=rubric,
        sensitivity=sensitivity,
        reasoning=data.get("reasoning", ""),
    )
