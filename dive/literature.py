"""Retrieval of real, verifiable published studies.

Dive's advisor does not recall literature - it retrieves it. Every study the
advisor is allowed to cite comes from a live query against Europe PMC (primary)
or PubMed E-utilities (fallback), and carries a real identifier that resolves.

This is the whole defence against fabricated citations. A language model asked
to "cite the evidence" will produce references that look correct in every
respect - plausible authors, a real journal, a well-formed DOI - and do not
exist. The only reliable fix is to never let the model supply the citation:
retrieve first, pass the retrieved records in, and then verify that every
identifier the model returned was one of the records you gave it.

``verify_identifiers`` is that final check, and it runs on every advisor
response.

Network note: both endpoints must be reachable. In restricted environments
retrieval fails cleanly and returns an empty result with a stated reason - the
advisor then declines to produce literature-grounded output rather than falling
back on recall.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Sequence

__all__ = [
    "Study",
    "LiteratureResult",
    "LiteratureClient",
    "EVIDENCE_PUB_TYPES",
]

EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: Publication types that carry validated, appraisable evidence. The advisor is
#: restricted to these - a narrative review or an editorial is not a basis for
#: an individual recommendation.
EVIDENCE_PUB_TYPES: tuple[str, ...] = (
    "systematic review",
    "meta-analysis",
    "randomized controlled trial",
    "clinical trial",
    "multicenter study",
    "observational study",
    "cohort studies",
)

#: Ranked for display and for the advisor's own weighting.
_PUB_TYPE_RANK = {
    "meta-analysis": 0,
    "systematic review": 1,
    "randomized controlled trial": 2,
    "clinical trial": 3,
    "multicenter study": 4,
    "observational study": 5,
    "cohort studies": 6,
}


@dataclass
class Study:
    """One retrieved publication. Every field comes from the source record."""

    identifier: str                 # canonical "PMID:12345678" or "DOI:10.x/y"
    title: str
    journal: str = ""
    year: str = ""
    authors: str = ""
    abstract: str = ""
    pub_types: tuple[str, ...] = ()
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    cited_by: int = 0
    source_api: str = "europepmc"

    @property
    def url(self) -> str:
        """Resolvable link, derived from the identifier when the field is unset.

        Records from the search APIs always populate ``pmid``/``doi``, but a
        Study reconstructed from a stored identifier alone must still produce a
        working link - a citation the researcher cannot click is not verifiable
        in any practical sense.
        """
        pmid = self.pmid
        doi = self.doi
        if not pmid and self.identifier.upper().startswith("PMID:"):
            pmid = self.identifier.split(":", 1)[1].strip()
        if not doi and self.identifier.upper().startswith("DOI:"):
            doi = self.identifier.split(":", 1)[1].strip()
        if pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if doi:
            return f"https://doi.org/{doi}"
        return ""

    @property
    def evidence_rank(self) -> int:
        ranks = [_PUB_TYPE_RANK.get(t.lower(), 9) for t in self.pub_types]
        return min(ranks) if ranks else 9

    def short_citation(self) -> str:
        first_author = self.authors.split(",")[0].strip() if self.authors else "Unknown"
        return f"{first_author} et al. {self.journal} {self.year}".strip()

    def as_prompt_record(self, abstract_chars: int = 1600) -> str:
        """Compact form given to the advisor. Abstract is trimmed, not dropped."""
        abstract = (self.abstract or "").strip()
        if len(abstract) > abstract_chars:
            abstract = abstract[:abstract_chars].rsplit(" ", 1)[0] + " [truncated]"
        return (
            f"[{self.identifier}]\n"
            f"Title: {self.title}\n"
            f"Journal: {self.journal} ({self.year})\n"
            f"Type: {', '.join(self.pub_types) or 'unspecified'}\n"
            f"Abstract: {abstract or 'not available'}"
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["pub_types"] = list(self.pub_types)
        payload["url"] = self.url
        return payload


@dataclass
class LiteratureResult:
    studies: list[Study] = field(default_factory=list)
    query: str = ""
    available: bool = True
    reason: str = ""
    source_api: str = ""

    def __len__(self) -> int:
        return len(self.studies)

    def identifiers(self) -> set[str]:
        return {s.identifier for s in self.studies}

    def by_identifier(self) -> dict[str, Study]:
        return {s.identifier: s for s in self.studies}


def _http_get_json(url: str, params: dict, timeout: float) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full, headers={"User-Agent": "Dive/1.0 (research tool)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class LiteratureClient:
    """Searches Europe PMC, falling back to PubMed."""

    def __init__(self, timeout: float = 20.0, email: str | None = None):
        self.timeout = timeout
        self.email = email

    # -- search -----------------------------------------------------------

    def search(
        self,
        terms: Sequence[str],
        max_results: int = 12,
        min_year: int | None = 2005,
        evidence_only: bool = True,
        extra_query: str = "",
    ) -> LiteratureResult:
        """Search for appraisable evidence on the given terms."""
        query = self._build_query(terms, min_year, evidence_only, extra_query)

        try:
            studies = self._search_europepmc(query, max_results)
            if studies:
                return LiteratureResult(studies, query, True, "", "europepmc")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            europe_error = f"Europe PMC unreachable ({type(exc).__name__}: {exc})"
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            europe_error = f"Europe PMC returned an unusable response ({exc})"
        else:
            europe_error = "Europe PMC returned no matching records"

        try:
            studies = self._search_pubmed(terms, max_results, min_year, evidence_only)
            if studies:
                return LiteratureResult(studies, query, True, "", "pubmed")
            reason = f"{europe_error}; PubMed returned no matching records."
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            reason = f"{europe_error}; PubMed unreachable ({type(exc).__name__}: {exc})."
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            reason = f"{europe_error}; PubMed returned an unusable response ({exc})."

        return LiteratureResult([], query, False, reason, "")

    def _build_query(
        self, terms: Sequence[str], min_year: int | None, evidence_only: bool, extra: str
    ) -> str:
        parts = [f'"{t}"' if " " in t else t for t in terms if t]
        query = " AND ".join(parts) if parts else "*"
        if extra:
            query = f"({query}) AND ({extra})"
        if min_year:
            query += f" AND (FIRST_PDATE:[{min_year}-01-01 TO 3000-01-01])"
        if evidence_only:
            types = " OR ".join(f'PUB_TYPE:"{t}"' for t in EVIDENCE_PUB_TYPES)
            query += f" AND ({types})"
        query += " AND (HAS_ABSTRACT:Y)"
        return query

    def _search_europepmc(self, query: str, max_results: int) -> list[Study]:
        payload = _http_get_json(
            EUROPE_PMC,
            {
                "query": query,
                "format": "json",
                "pageSize": min(max_results, 25),
                "resultType": "core",
                "sort": "CITED desc",
            },
            self.timeout,
        )
        records = payload.get("resultList", {}).get("result", [])
        studies: list[Study] = []
        for record in records:
            pmid = str(record.get("pmid", "") or "")
            doi = str(record.get("doi", "") or "")
            if not pmid and not doi:
                continue
            pub_types = tuple(
                t.lower() for t in (record.get("pubTypeList", {}) or {}).get("pubType", []) or []
            )
            studies.append(
                Study(
                    identifier=f"PMID:{pmid}" if pmid else f"DOI:{doi}",
                    title=str(record.get("title", "")).strip().rstrip("."),
                    journal=str(record.get("journalTitle", "") or ""),
                    year=str(record.get("pubYear", "") or ""),
                    authors=str(record.get("authorString", "") or ""),
                    abstract=str(record.get("abstractText", "") or ""),
                    pub_types=pub_types,
                    doi=doi,
                    pmid=pmid,
                    pmcid=str(record.get("pmcid", "") or ""),
                    cited_by=int(record.get("citedByCount", 0) or 0),
                    source_api="europepmc",
                )
            )
        studies.sort(key=lambda s: (s.evidence_rank, -s.cited_by))
        return studies[:max_results]

    def _search_pubmed(
        self, terms: Sequence[str], max_results: int, min_year: int | None, evidence_only: bool
    ) -> list[Study]:
        query = " AND ".join(f'"{t}"[All Fields]' for t in terms if t)
        if evidence_only:
            filters = " OR ".join(f'"{t}"[Publication Type]' for t in EVIDENCE_PUB_TYPES)
            query += f" AND ({filters})"
        if min_year:
            query += f' AND ("{min_year}"[Date - Publication] : "3000"[Date - Publication])'

        params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
        if self.email:
            params["email"] = self.email
        found = _http_get_json(PUBMED_SEARCH, params, self.timeout)
        pmids = found.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        time.sleep(0.34)  # NCBI asks for <=3 requests/second without a key
        summary = _http_get_json(
            PUBMED_SUMMARY,
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
            self.timeout,
        )
        result = summary.get("result", {})

        studies: list[Study] = []
        for pmid in pmids:
            record = result.get(pmid)
            if not record:
                continue
            authors = ", ".join(a.get("name", "") for a in record.get("authors", [])[:6])
            doi = next(
                (i.get("value", "") for i in record.get("articleids", [])
                 if i.get("idtype") == "doi"), ""
            )
            studies.append(
                Study(
                    identifier=f"PMID:{pmid}",
                    title=str(record.get("title", "")).strip().rstrip("."),
                    journal=str(record.get("fulljournalname", "") or record.get("source", "")),
                    year=str(record.get("pubdate", ""))[:4],
                    authors=authors,
                    abstract="",  # esummary carries no abstract; title+type still verifiable
                    pub_types=tuple(t.lower() for t in record.get("pubtype", []) or []),
                    doi=doi,
                    pmid=pmid,
                    source_api="pubmed",
                )
            )
        studies.sort(key=lambda s: s.evidence_rank)
        return studies

    # -- verification -----------------------------------------------------

    @staticmethod
    def verify_identifiers(
        claimed: Iterable[str], retrieved: LiteratureResult
    ) -> tuple[list[str], list[str]]:
        """Split claimed identifiers into (verified, unverifiable).

        An identifier is verified only if it was in the retrieved set handed to
        the model. Anything else - however well formed - is discarded. This is
        the check that makes a fabricated citation impossible to surface.
        """
        available = retrieved.identifiers()
        normalised = {a.upper().replace(" ", ""): a for a in available}
        verified: list[str] = []
        unverifiable: list[str] = []
        for item in claimed:
            key = str(item).upper().replace(" ", "")
            if key in normalised:
                verified.append(normalised[key])
            else:
                unverifiable.append(str(item))
        return verified, unverifiable
