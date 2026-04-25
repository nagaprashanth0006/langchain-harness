"""Academic API retrievers: PubMed, Semantic Scholar, OpenAlex, Crossref, arXiv."""
from __future__ import annotations
import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from ..config import get_config
from ..models import Source, SourceType

logger = logging.getLogger(__name__)

_TIMEOUT = 15


async def _get(url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict | str]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=headers or {})
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return resp.json()
                return resp.text
    except Exception as e:
        logger.debug(f"GET {url} failed: {e}")
    return None


# ── PubMed ─────────────────────────────────────────────────────────────────

async def _pubmed_search(query: str, max_results: int = 5) -> List[str]:
    cfg = get_config()
    base = cfg["datasources"]["pubmed"]["base_url"]
    api_key = cfg["datasources"]["pubmed"].get("api_key") or ""
    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    data = await _get(f"{base}/esearch.fcgi", params=params)
    if isinstance(data, dict):
        return data.get("esearchresult", {}).get("idlist", [])
    return []


async def _pubmed_fetch(pmids: List[str]) -> List[Source]:
    if not pmids:
        return []
    cfg = get_config()
    base = cfg["datasources"]["pubmed"]["base_url"]
    api_key = cfg["datasources"]["pubmed"].get("api_key") or ""
    params: Dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    xml_text = await _get(f"{base}/efetch.fcgi", params=params)
    if not xml_text or not isinstance(xml_text, str):
        return []

    sources = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            title_el = article.find(".//ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""
            abstract_el = article.find(".//AbstractText")
            abstract = abstract_el.text or "" if abstract_el is not None else ""
            pub_type = "peer-reviewed"
            pt_el = article.find(".//PublicationType")
            if pt_el is not None and "preprint" in (pt_el.text or "").lower():
                pub_type = "preprint"
            src = Source(
                origin=["pubmed"],
                title=title,
                text=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                type=SourceType(pub_type) if pub_type in SourceType._value2member_map_ else SourceType.PEER_REVIEWED,
            )
            sources.append(src)
    except ET.ParseError as e:
        logger.warning(f"PubMed XML parse error: {e}")
    return sources


async def pubmed_retrieve(query: str, max_results: int = 5) -> List[Source]:
    pmids = await _pubmed_search(query, max_results)
    return await _pubmed_fetch(pmids)


# ── Semantic Scholar ────────────────────────────────────────────────────────

async def semantic_scholar_retrieve(query: str, max_results: int = 5) -> List[Source]:
    cfg = get_config()
    base = cfg["datasources"]["semantic_scholar"]["base_url"]
    api_key = cfg["datasources"]["semantic_scholar"].get("api_key") or ""
    headers = {"x-api-key": api_key} if api_key else {}
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,url,year,isOpenAccess,publicationTypes",
    }
    data = await _get(f"{base}/paper/search", params=params, headers=headers)
    if not isinstance(data, dict):
        return []
    sources = []
    for paper in data.get("data", []):
        abstract = paper.get("abstract") or ""
        title = paper.get("title") or ""
        url = paper.get("url") or ""
        pub_types = paper.get("publicationTypes") or []
        is_preprint = any("preprint" in (t or "").lower() for t in pub_types)
        src = Source(
            origin=["semantic_scholar"],
            title=title,
            text=abstract,
            url=url,
            type=SourceType.PREPRINT if is_preprint else SourceType.PEER_REVIEWED,
        )
        sources.append(src)
    return sources


# ── OpenAlex ────────────────────────────────────────────────────────────────

async def openalexs_retrieve(query: str, max_results: int = 5) -> List[Source]:
    cfg = get_config()
    base = cfg["datasources"]["openalexs"]["base_url"]
    email = cfg["datasources"]["openalexs"].get("email") or ""
    params: Dict[str, Any] = {
        "search": query,
        "per-page": max_results,
        "select": "title,abstract_inverted_index,doi,primary_location,type",
    }
    if email:
        params["mailto"] = email
    data = await _get(f"{base}/works", params=params)
    if not isinstance(data, dict):
        return []
    sources = []
    for work in data.get("results", []):
        title = work.get("title") or ""
        doi = work.get("doi") or ""
        url = f"https://doi.org/{doi}" if doi else ""
        inv_index = work.get("abstract_inverted_index") or {}
        abstract = _reconstruct_abstract(inv_index)
        work_type = work.get("type") or "article"
        src = Source(
            origin=["openalexs"],
            title=title,
            text=abstract,
            url=url,
            type=SourceType.PEER_REVIEWED if "journal" in work_type else SourceType.ARTICLE,
        )
        sources.append(src)
    return sources


def _reconstruct_abstract(inverted_index: Dict[str, List[int]]) -> str:
    if not inverted_index:
        return ""
    pairs = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


# ── Crossref ────────────────────────────────────────────────────────────────

async def crossref_retrieve(query: str, max_results: int = 5) -> List[Source]:
    cfg = get_config()
    base = cfg["datasources"]["crossref"]["base_url"]
    email = cfg["datasources"]["crossref"].get("email") or ""
    params: Dict[str, Any] = {
        "query": query,
        "rows": max_results,
        "select": "title,abstract,DOI,URL,type",
    }
    if email:
        params["mailto"] = email
    data = await _get(f"{base}/works", params=params)
    if not isinstance(data, dict):
        return []
    sources = []
    for item in data.get("message", {}).get("items", []):
        title_parts = item.get("title") or []
        title = title_parts[0] if title_parts else ""
        abstract = item.get("abstract") or ""
        url = item.get("URL") or ""
        src = Source(
            origin=["crossref"],
            title=title,
            text=abstract,
            url=url,
            type=SourceType.PEER_REVIEWED,
        )
        sources.append(src)
    return sources


# ── arXiv ───────────────────────────────────────────────────────────────────

async def arxiv_retrieve(query: str, max_results: int = 5) -> List[Source]:
    cfg = get_config()
    base = cfg["datasources"]["arxiv"]["base_url"]
    params = {
        "search_query": f"all:{quote_plus(query)}",
        "max_results": max_results,
    }
    xml_text = await _get(base, params=params)
    if not xml_text or not isinstance(xml_text, str):
        return []
    sources = []
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary_el = entry.find("atom:summary", ns)
            abstract = (summary_el.text or "").strip() if summary_el is not None else ""
            link_el = entry.find("atom:id", ns)
            url = (link_el.text or "").strip() if link_el is not None else ""
            src = Source(
                origin=["arxiv"],
                title=title,
                text=abstract,
                url=url,
                type=SourceType.PREPRINT,
            )
            sources.append(src)
    except ET.ParseError as e:
        logger.warning(f"arXiv parse error: {e}")
    return sources


# ── Dispatcher ──────────────────────────────────────────────────────────────

_RETRIEVERS = {
    "pubmed": pubmed_retrieve,
    "semantic_scholar": semantic_scholar_retrieve,
    "openalexs": openalexs_retrieve,
    "crossref": crossref_retrieve,
    "arxiv": arxiv_retrieve,
}


async def retrieve_from_academic(
    query: str,
    sources: List[str],
    max_per_source: int = 5,
) -> List[Source]:
    tasks = []
    for src_id in sources:
        fn = _RETRIEVERS.get(src_id)
        if fn:
            tasks.append(fn(query, max_per_source))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_sources: List[Source] = []
    for r in results:
        if isinstance(r, list):
            all_sources.extend(r)
        elif isinstance(r, Exception):
            logger.warning(f"Academic retriever error: {r}")
    return all_sources
