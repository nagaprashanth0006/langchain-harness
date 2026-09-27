"Prompt templates for RAG generation."

from __future__ import annotations

from dataclasses import dataclass
from models import ScoredChunk


@dataclass
class RAGPromptTemplate:
    """Configurable prompt template for RAG generation."""

    system: str = (
        "You are a precise and helpful assistant. "
        "Answer the user's question using ONLY the provided context passages. "
        "If the answer is not present in the context, explicitly say "
        "'I cannot find this information in the provided documents.' "
        "Cite the source of each claim using the format [Source N]. "
        "Do not add information from your training data unless explicitly asked."
    )

    context_header: str = "CONTEXT PASSAGES:"
    context_item_template: str = "[Source {n}] ({source}, page {page}):\n{content}"
    question_header: str = "QUESTION:"
    answer_instruction: str = (
        "Provide a comprehensive answer grounded in the context above. "
        "Use inline citations [Source N] to reference specific passages."
    )

    unknown_source: str = "unknown"
    unknown_page: str = "N/A"


_DEFAULT_TEMPLATE = RAGPromptTemplate()


def format_context(
    scored_chunks: list[ScoredChunk],
    template: RAGPromptTemplate | None = None,
) -> str:
    """Format retrieved chunks into a numbered context block."""
    tmpl = template or _DEFAULT_TEMPLATE
    parts: list[str] = [tmpl.context_header, ""]

    for i, sc in enumerate(scored_chunks, start=1):
        source = (
            sc.chunk.metadata.get("file_name") or sc.chunk.source or tmpl.unknown_source
        )
        page = sc.chunk.metadata.get("page") or tmpl.unknown_page

        passage = tmpl.context_item_template.format(
            n=i,
            source=source,
            page=page,
            content=sc.chunk.content.strip(),
        )
        parts.append(passage)
        parts.append("")

    return "\n".join(parts)


def build_rag_prompt(
    query: str,
    scored_chunks: list[ScoredChunk],
    template: RAGPromptTemplate | None = None,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for the generation LLM.

    Returns:
        (system_prompt, user_message) — both are strings.
        Pass system_prompt to the system role, user_message to the user role.
    """
    tmpl = template or _DEFAULT_TEMPLATE
    context_block = format_context(scored_chunks, tmpl)

    user_message = (
        f"{context_block}\n{tmpl.question_header}\n{query}\n\n{tmpl.answer_instruction}"
    )

    return tmpl.system, user_message


def build_hypothesis_prompt(query: str) -> str:
    """
    Prompt to generate a HyDE (hypothetical document) for retrieval.
    The hypothetical answer is used for embedding-based retrieval, NOT shown to user.
    """
    return (
        f"Write a short, factual passage that would answer the following question. "
        f"Be specific and detailed. This will be used for document retrieval.\n\n"
        f"Question: {query}\n\n"
        f"Hypothetical answer passage:"
    )
