"""The advisor must be structurally incapable of surfacing an invented citation."""

import json
from types import SimpleNamespace

import pytest

from dive.advisor import Advisor, AdvisorResult, ParticipantBrief, brief_from_profile
from dive.literature import LiteratureClient, LiteratureResult, Study

REAL = [
    Study(identifier="PMID:19588334", title="Progressive resistance strength training",
          journal="Cochrane Database Syst Rev", year="2009",
          pub_types=("systematic review", "meta-analysis")),
    Study(identifier="PMID:30703272", title="Exercise for preventing falls",
          journal="Cochrane Database Syst Rev", year="2019",
          pub_types=("systematic review",)),
]
RETRIEVED = LiteratureResult(studies=list(REAL), query="q", available=True)


class _FakeResponse:
    def __init__(self, payload):
        self.content = [SimpleNamespace(type="text", text=json.dumps(payload))]
        self.stop_reason = "end_turn"
        self.usage = SimpleNamespace(input_tokens=100, output_tokens=200)


def _verify(payload) -> AdvisorResult:
    return Advisor()._verify(payload, RETRIEVED, _FakeResponse(payload))


def test_a_fabricated_citation_is_discarded():
    result = _verify({
        "recommendations": [{
            "title": "Progressive resistance training",
            "rationale": "Improves strength.",
            "specific_to_participant": "Grip 13.5 kg is below the female cutoff.",
            "evidence_strength": "strong",
            "applicability_caveat": "Trials enrolled younger participants.",
            # one real, one plausible-looking invention
            "citation_ids": ["PMID:19588334", "PMID:28472115"],
        }],
        "evidence_gaps": [],
    })
    assert len(result.recommendations) == 1
    recommendation = result.recommendations[0]
    assert recommendation.citation_ids == ["PMID:19588334"]
    assert recommendation.dropped_citations == ["PMID:28472115"]
    assert result.n_citations_dropped == 1


def test_a_recommendation_with_no_verifiable_source_is_dropped_entirely():
    result = _verify({
        "recommendations": [{
            "title": "Invented intervention",
            "rationale": "Claimed benefit.",
            "specific_to_participant": "Applies here.",
            "evidence_strength": "strong",
            "applicability_caveat": "None stated.",
            "citation_ids": ["PMID:99999999", "DOI:10.1000/made.up"],
        }],
        "evidence_gaps": [],
    })
    assert result.recommendations == []
    assert result.n_citations_dropped == 2


def test_verified_citations_resolve_to_the_retrieved_studies():
    result = _verify({
        "recommendations": [{
            "title": "Balance and functional exercise",
            "rationale": "Reduces falls.",
            "specific_to_participant": "Gait speed 0.62 m/s.",
            "evidence_strength": "strong",
            "applicability_caveat": "Community-dwelling samples.",
            "citation_ids": ["PMID:30703272"],
        }],
        "evidence_gaps": ["No trial covers the B/T ratio."],
    })
    recommendation = result.recommendations[0]
    assert [s.title for s in recommendation.studies] == ["Exercise for preventing falls"]
    assert recommendation.studies[0].url == "https://pubmed.ncbi.nlm.nih.gov/30703272/"
    assert result.evidence_gaps == ["No trial covers the B/T ratio."]


def test_identifier_verification_is_whitespace_and_case_insensitive():
    verified, unverifiable = LiteratureClient.verify_identifiers(
        ["pmid: 19588334", "PMID:19588334"], RETRIEVED
    )
    assert verified == ["PMID:19588334", "PMID:19588334"]
    assert unverifiable == []


def test_advisor_declines_when_no_literature_could_be_retrieved():
    class _Offline:
        def search(self, *args, **kwargs):
            return LiteratureResult([], "q", available=False, reason="network blocked")

    advisor = Advisor(literature_client=_Offline())
    available, _ = Advisor.client_available()
    if not available:
        pytest.skip("Anthropic credential not configured in this environment")

    result = advisor.advise(ParticipantBrief(findings=["low_grip"]))
    assert not result.available
    assert "recall" in result.reason


def test_advisor_reports_unavailable_without_a_credential(monkeypatch):
    import anthropic

    def _explode(*args, **kwargs):
        raise anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)

    monkeypatch.setattr(anthropic, "Anthropic", _explode)
    available, reason = Advisor.client_available()
    assert not available
    assert "credential" in reason


def test_brief_contains_only_measured_values():
    profile = {"sex": 1.0, "age_years": 79, "max_grip_kg": 13.5, "gait_speed_ms": 0.62,
               "sppb_total": 7, "fried_score": 3}
    brief = brief_from_profile(profile)
    block = brief.as_prompt_block()

    assert "13.5" in block and "0.62" in block
    assert brief.sex_label == "female"
    # Findings drive retrieval and must come from threshold logic, not free text.
    assert "low_grip" in brief.findings
    assert brief.search_terms()


def test_brief_marks_unmeasured_variables_as_unknown():
    brief = brief_from_profile({"sex": 1.0, "age_years": 75})
    block = brief.as_prompt_block()
    assert "NOT MEASURED" in block
    assert "do not assume these are normal" in block
