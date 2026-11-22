import json

import pytest

from analytics_agent.agents.data_quality_reviewer import (DataQualityReview, DataQualityReviewer, ReviewFinding,
                                                          build_prompt, build_review_payload, check_review,
                                                          deterministic_review)
from analytics_agent.data_quality import validate_tables
from analytics_agent.llm import GeminiStructuredClient, LLMError, classify_error
from analytics_agent.testing import (ScriptedGeminiClient, fake_api_error, fake_connection_error,
                                    fake_models_api_error)


@pytest.fixture(scope="module")
def rep(corrupted, cfg):
    return validate_tables(corrupted, cfg.as_of_ts, "corrupted")


@pytest.fixture(scope="module")
def clean_rep(clean, cfg):
    return validate_tables(clean, cfg.as_of_ts, "clean")


def good_json(rep):
    return deterministic_review(rep).model_dump_json()


def llm(script, **kw):
    return GeminiStructuredClient("gemini-test", client=ScriptedGeminiClient(script), sleep=lambda s: None, **kw)


# ---- deterministic reviewer ----
def test_deterministic_review_passes_guardrails(rep, clean_rep):
    for r in (rep, clean_rep):
        g = check_review(deterministic_review(r), r)
        assert g.passed, g.violations
    assert deterministic_review(rep).overall_assessment == "not_fit_for_use"
    assert deterministic_review(clean_rep).overall_assessment == "fit_for_use"


def test_payload_minimal_and_redacted(rep):
    p = build_review_payload(rep)
    assert len(p["non_passing_rules"]) == 20                       # passing rules not sent
    text = json.dumps(p)
    assert "@" not in text and "email#" in text                      # PII masked
    assert "rows" not in p                                           # no raw records


# ---- guardrails ----
def _mutate(rep, **changes):
    r = deterministic_review(rep).model_copy(deep=True)
    for k, v in changes.items():
        setattr(r, k, v)
    return r


def test_guardrail_rejects_hallucinated_rule(rep):
    r = _mutate(rep)
    r.findings[0].rule_ids.append("CUS_SPACE_ALIENS")
    assert any("unknown rule" in v for v in check_review(r, rep).violations)


def test_guardrail_rejects_dropped_failure(rep):
    r = _mutate(rep)
    r.findings = r.findings[1:]
    assert any("not covered" in v for v in check_review(r, rep).violations)


def test_guardrail_rejects_severity_downgrade(rep):
    r = _mutate(rep)
    crit = next(f for f in r.findings if f.severity == "critical")
    crit.severity = "low"
    assert any("lower than deterministic severity" in v for v in check_review(r, rep).violations)


def test_guardrail_rejects_fit_for_use_on_bad_data(rep):
    r = _mutate(rep, overall_assessment="fit_for_use")
    assert any("overall_assessment" in v for v in check_review(r, rep).violations)


def test_guardrail_rejects_citing_passing_rule(rep):
    r = _mutate(rep)
    r.findings[0].rule_ids.append("PRD_PK_UNIQUE")
    assert any("PASSED" in v for v in check_review(r, rep).violations)


def test_guardrail_requires_evidence_counts(rep):
    r = _mutate(rep)
    r.findings[0].evidence = ["Something looks wrong."]
    assert any("observed_count" in v for v in check_review(r, rep).violations)


# ---- orchestration with scripted LLM ----
def test_accepted_first_try(rep):
    res = DataQualityReviewer(llm([good_json(rep)])).review(rep)
    assert res.outcome == "accepted" and res.reviewer == "gemini:gemini-test" and res.guardrail.passed


def test_revised_after_rejection_feeds_back_violations(rep):
    bad = _mutate(rep, overall_assessment="fit_for_use").model_dump_json()
    fake = ScriptedGeminiClient([bad, good_json(rep)])
    res = DataQualityReviewer(GeminiStructuredClient("m", client=fake)).review(rep)
    assert res.outcome == "revised" and res.attempts == 2
    assert "REJECTED" in fake.calls[1]["input"] and "overall_assessment" in fake.calls[1]["input"]


def test_escalates_after_bounded_revisions(rep):
    bad = _mutate(rep, overall_assessment="fit_for_use").model_dump_json()
    fake = ScriptedGeminiClient([bad, bad, bad])
    res = DataQualityReviewer(GeminiStructuredClient("m", client=fake), max_revisions=1).review(rep)
    assert res.outcome == "escalated" and res.reviewer == "deterministic"
    assert len(fake.calls) == 2 and res.requires_human_review          # bounded: 1 try + 1 revision


def test_invalid_json_then_valid(rep):
    res = DataQualityReviewer(llm(["{not json", good_json(rep)])).review(rep)
    assert res.outcome == "revised"


def test_abstains_without_calling_llm(clean_rep):
    fake = ScriptedGeminiClient([])
    res = DataQualityReviewer(GeminiStructuredClient("m", client=fake)).review(clean_rep)
    assert res.outcome == "abstained" and fake.calls == []


def test_test_mode_without_llm(rep):
    res = DataQualityReviewer(None).review(rep)
    assert res.outcome == "deterministic_only" and res.guardrail.passed


# ---- LLM client: retries and error classification ----
def test_retries_rate_limit_then_succeeds(rep):
    c = llm([fake_api_error(429, "RESOURCE_EXHAUSTED", "Rate limit exceeded, retry later"), good_json(rep)])
    review, meta = c.generate(system_instruction="s", prompt="p", schema=DataQualityReview)
    assert meta.attempts == 2 and meta.errors == ["rate_limit"]


def test_retries_are_bounded(rep):
    errs = [fake_api_error(503, "UNAVAILABLE", "overloaded")] * 5
    c = llm(errs, max_retries=2)
    with pytest.raises(LLMError) as e:
        c.generate(system_instruction="s", prompt="p", schema=DataQualityReview)
    assert e.value.kind == "unavailable" and e.value.attempts == 3


@pytest.mark.parametrize("code,status,msg,kind", [
    (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "auth"),
    (403, "PERMISSION_DENIED", "permission denied", "auth"),
    (404, "NOT_FOUND", "models/gemini-x is not found", "model_not_found"),
    (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota", "quota"),
    (429, "RESOURCE_EXHAUSTED", "Rate limit: requests per minute", "rate_limit"),
    (400, "INVALID_ARGUMENT", "schema too complex", "invalid_request"),
    (500, "INTERNAL", "internal", "unavailable"),
])
def test_error_classification(code, status, msg, kind):
    # Interactions API exception family (what client.interactions.create raises)
    assert classify_error(fake_api_error(code, status, msg)).kind == kind
    # google.genai.errors family (client.models.*) is handled the same way
    assert classify_error(fake_models_api_error(code, status, msg)).kind == kind


def test_interactions_errors_are_real_sdk_types():
    e = fake_api_error(429, "RESOURCE_EXHAUSTED", "Rate limit exceeded")
    assert type(e).__name__ == "RateLimitError" and e.status_code == 429
    assert type(fake_connection_error()).__name__ == "APIConnectionError"


def test_sdk_connection_and_timeout_errors():
    assert classify_error(fake_connection_error()).kind == "network"
    assert classify_error(fake_connection_error(timeout=True)).kind == "timeout"


def test_connection_error_is_retried(rep):
    c = llm([fake_connection_error(), good_json(rep)])
    _, meta = c.generate(system_instruction="s", prompt="p", schema=DataQualityReview)
    assert meta.errors == ["network"] and meta.attempts == 2


def test_quota_and_auth_fail_fast(rep):
    fake = ScriptedGeminiClient([fake_api_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota")])
    c = GeminiStructuredClient("m", client=fake, sleep=lambda s: None)
    with pytest.raises(LLMError) as e:
        c.generate(system_instruction="s", prompt="p", schema=DataQualityReview)
    assert e.value.kind == "quota" and len(fake.calls) == 1


def test_network_error_classified():
    import httpx
    assert classify_error(httpx.ConnectError("no route")).kind == "network"
    assert classify_error(httpx.ReadTimeout("slow")).kind == "timeout"


def test_truncated_output_is_invalid(rep):
    c = llm([("incomplete", '{"overall_assessment": "not_fit')])
    with pytest.raises(LLMError) as e:
        c.generate(system_instruction="s", prompt="p", schema=DataQualityReview)
    assert e.value.kind == "invalid_output"


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError) as e:
        GeminiStructuredClient("m")
    assert e.value.kind == "not_configured"


def test_request_uses_structured_output(rep):
    fake = ScriptedGeminiClient([good_json(rep)])
    GeminiStructuredClient("gemini-x", client=fake).generate(system_instruction="sys", prompt=build_prompt(rep),
                                                             schema=DataQualityReview)
    call = fake.calls[0]
    assert call["model"] == "gemini-x" and call["system_instruction"] == "sys"
    assert call["response_format"]["mime_type"] == "application/json"
    assert "overall_assessment" in json.dumps(call["response_format"]["schema"])
    assert "<dq_report>" in call["input"] and "@example.com" not in call["input"]
