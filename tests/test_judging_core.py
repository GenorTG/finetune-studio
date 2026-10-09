"""Core of the run-then-judge design: a run records raw transcripts and no verdict; judging is a separate step.

Pins: nothing on the custom-dataset path compares strings; the judge reply parser never guesses a verdict from
prose; a failed judge call leaves a case unjudged (never fail); a human verdict beats an AI verdict and can be
retracted; scores keep unjudged cases as "awaiting".
"""

from __future__ import annotations

import json

import pytest

from finetune_studio import db
from finetune_studio.db import judgements as jdb
from finetune_studio.testing import judge as judge_mod
from finetune_studio.testing.judge import (
    JudgeCase,
    LoadedJudge,
    build_judge_messages,
    judge_many,
    judge_one,
    parse_judge_reply,
)
from finetune_studio.testing.judging import judge_benchmark
from finetune_studio.testing.scoring import rescore_benchmark
from finetune_studio.testing.suite import (
    BenchmarkCase,
    CaseResult,
    run_suite,
    score_results,
)


class FakeEngine:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, messages, **_kw):
        return self.answers[messages[-1]["content"]]


def _loaded(chat, *, concurrent=False) -> LoadedJudge:
    return LoadedJudge(chat=chat, provider_id="fake", model="fake-judge", label="Fake judge", concurrent=concurrent)


def _saved_run(answers: list[tuple[str, str, str]]) -> tuple[str, list[str]]:
    """A saved test run (question, key, model answer per case) with no verdicts. Returns (benchmark id, case ids)."""
    pid = db.create_project(name="p")["id"]
    rid = db.create_run(pid, "run")["id"]
    bench = db.create_benchmark(rid, "quiz", {}, status="running", kind="suite")
    ids = [db.create_case(bench["id"], rid, f"c{i}", "general", q, key, ans, [{"role": "user", "content": q}])
           for i, (q, key, ans) in enumerate(answers)]
    return bench["id"], ids


# ── run: raw transcripts only ────────────────────────────────────────────────


def test_run_suite_records_transcripts_and_never_a_verdict():
    cases = [BenchmarkCase(name="a", question="Q1", correct_answer="42", keywords=["42"])]
    results = run_suite(FakeEngine({"Q1": "It is forty-two."}), cases)
    assert results[0].model_answer == "It is forty-two."
    assert results[0].verdict == "" and results[0].judge == "none"
    assert results[0].transcript[-1] == {"role": "assistant", "content": "It is forty-two."}
    assert score_results(results)["pass_rate"] is None  # nothing judged: no rate, not 0 %


def test_run_suite_streams_each_result_and_can_stop():
    seen: list[str] = []
    cases = [BenchmarkCase(name=f"c{i}", question=f"Q{i}", correct_answer="x") for i in range(4)]
    engine = FakeEngine({f"Q{i}": "a" for i in range(4)})
    out = run_suite(engine, cases, on_result=lambda r: seen.append(r.case_name), should_stop=lambda: len(seen) >= 2)
    assert seen == ["c0", "c1"] and len(out) == 2


def test_a_failing_generation_is_recorded_as_an_error_not_a_verdict():
    class Boom:
        def generate(self, *_a, **_k):
            raise RuntimeError("cuda oom")

    r = run_suite(Boom(), [BenchmarkCase(name="a", question="Q", correct_answer="x")])[0]
    assert r.error == "cuda oom" and r.verdict == ""


def test_score_results_counts_awaiting_against_total():
    judged = CaseResult("a", "g", "q", "k", "a", verdict="pass", judge="ai")
    waiting = CaseResult("b", "g", "q", "k", "a")
    s = score_results([judged, waiting])
    assert (s["total"], s["judged"], s["awaiting"], s["pass_rate"]) == (2, 1, 1, 50.0)
    assert s["by_judge"] == {"ai": 1}


# ── judge reply parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [
    '{"reasoning": "has 42", "verdict": "pass", "confidence": 0.9}',
    '```json\n{"reasoning": "has 42", "verdict": "PASS", "confidence": 0.9}\n```',
    '<think>hmm {"verdict":"fail"}</think>\n{"reasoning": "has 42", "verdict": "pass"}',
    'Sure!\n{"reasoning": "answer {42} found", "verdict": "pass", "confidence": "high"}',
])
def test_parse_accepts_json_in_the_usual_wrappers(raw):
    res = parse_judge_reply(raw)
    assert res.verdict == "pass" and not res.error


def test_parse_clamps_confidence_and_keeps_reasoning():
    res = parse_judge_reply('{"reasoning": "ok", "verdict": "partial", "confidence": 7}')
    assert (res.verdict, res.reasoning, res.confidence) == ("partial", "ok", 1.0)


@pytest.mark.parametrize("raw", ["", "The answer looks correct to me, I would say pass.", '{"verdict": "maybe"}', "{not json"])
def test_parse_never_guesses_a_verdict_from_prose(raw):
    res = parse_judge_reply(raw)
    assert res.verdict == "" and res.error


def test_parse_reads_an_explicit_verdict_line():
    assert parse_judge_reply("Reasoning: fine.\nVerdict: partial").verdict == "partial"


# ── judge_one ────────────────────────────────────────────────────────────────


def test_judge_one_retries_a_malformed_reply_once():
    replies = iter(["no idea", '{"reasoning": "ok", "verdict": "pass"}'])
    res = judge_one(lambda _m: next(replies), JudgeCase("Q", "key", "answer"))
    assert res.verdict == "pass"


def test_judge_one_failure_is_an_error_never_a_fail():
    def boom(_m):
        raise ConnectionError("down")

    res = judge_one(boom, JudgeCase("Q", "key", "answer"))
    assert res.verdict == "" and "down" in res.error


def test_judge_prompt_carries_the_whole_answer_key_and_the_unanswerable_flag():
    msgs = build_judge_messages(JudgeCase("Q?", "the key", "model said", keywords=["a", "b"]))
    user = msgs[1]["content"]
    assert "the key" in user and "- a" in user and "- b" in user and "model said" in user and "UNANSWERABLE" not in user
    abst = build_judge_messages(JudgeCase("Q?", "", "I don't know", expect_abstain=True))[1]["content"]
    assert "UNANSWERABLE" in abst and "KEY VALUES" not in abst


# ── judging a saved run ──────────────────────────────────────────────────────


def _reply(verdict: str) -> str:
    return json.dumps({"reasoning": f"r-{verdict}", "verdict": verdict, "confidence": 0.8})


def test_judge_benchmark_appends_judgements_and_rescoring_follows():
    bid, ids = _saved_run([("Q1", "Paris", "The capital is Paris."), ("Q2", "Rome", "Madrid")])
    replies = iter([_reply("pass"), _reply("fail")])
    summary = judge_benchmark(bid, _loaded(lambda _m: next(replies)))
    assert (summary.judged, summary.failed) == (2, 0)
    cases = {c["id"]: c for c in db.list_cases(bid)}
    assert cases[ids[0]]["verdict"] == "pass" and cases[ids[0]]["judge"] == "ai" and cases[ids[0]]["judge_model"] == "fake-judge"
    assert cases[ids[1]]["verdict"] == "fail"
    scores = db.get_benchmark(bid)["scores"]
    assert scores["passed"] == 1 and scores["pass_rate"] == 50.0 and scores["by_judge"] == {"ai": 2}


def test_a_failed_judge_call_leaves_the_case_awaiting():
    bid, ids = _saved_run([("Q1", "a", "a"), ("Q2", "b", "b")])
    calls = iter([_reply("pass"), None])

    def chat(_m):
        r = next(calls)
        if r is None:
            raise TimeoutError("judge timed out")
        return r

    summary = judge_benchmark(bid, _loaded(chat))
    assert (summary.judged, summary.failed) == (1, 1)
    cases = {c["id"]: c for c in db.list_cases(bid)}
    assert cases[ids[1]]["verdict"] == "" and cases[ids[1]]["judge"] == "none"
    failed = [j for j in jdb.list_judgements(case_id=ids[1])]
    assert failed and failed[0]["verdict"] == "" and "timed out" in failed[0]["error"]
    assert db.get_benchmark(bid)["scores"]["awaiting"] == 1


def test_cases_whose_model_run_errored_are_not_sent_to_the_judge():
    pid = db.create_project(name="p")["id"]
    rid = db.create_run(pid, "run")["id"]
    bid = db.create_benchmark(rid, "quiz", {})["id"]
    db.create_case(bid, rid, "boom", "g", "Q", "key", "", [], error="cuda oom")
    asked: list[str] = []
    summary = judge_benchmark(bid, _loaded(lambda m: asked.append(m[-1]["content"]) or _reply("pass")))
    assert summary.skipped_run_error == 1 and not asked


def test_only_unjudged_keeps_human_and_ai_verdicts_but_rejudges_legacy_scripted_ones():
    bid, ids = _saved_run([("Q1", "k", "a"), ("Q2", "k", "a"), ("Q3", "k", "a")])
    jdb.add_judgement(ids[0], kind="human", verdict="fail", reasoning="mine")
    jdb.add_judgement(ids[1], kind="ai", verdict="pass", judge_model="old")
    jdb.add_judgement(ids[2], kind="scripted", verdict="fail", judge_model="heuristic")
    asked: list[str] = []
    judge_benchmark(bid, _loaded(lambda m: asked.append(m[-1]["content"]) or _reply("pass")), only_unjudged=True)
    assert len(asked) == 1 and "Q3" in asked[0]
    by_id = {c["id"]: c for c in db.list_cases(bid)}
    assert by_id[ids[2]]["verdict"] == "pass" and by_id[ids[2]]["judge"] == "ai"  # AI beats the legacy matcher


def test_rejudging_everything_still_lets_a_human_verdict_win():
    bid, ids = _saved_run([("Q1", "k", "a")])
    jdb.add_judgement(ids[0], kind="human", verdict="fail", reasoning="mine")
    judge_benchmark(bid, _loaded(lambda _m: _reply("pass")), only_unjudged=False)
    case = db.list_cases(bid)[0]
    assert case["verdict"] == "fail" and case["judge"] == "human"
    assert len(jdb.list_judgements(case_id=ids[0])) == 2  # the AI opinion is kept for comparison


def test_judge_many_parallel_for_api_providers_keeps_every_result():
    items = [(str(i), JudgeCase(f"Q{i}", "k", "a")) for i in range(11)]
    got: dict[str, str] = {}
    judge_many(_loaded(lambda _m: _reply("pass"), concurrent=True), items, lambda k, r: got.__setitem__(k, r.verdict))
    assert got == {str(i): "pass" for i in range(11)}


def test_stop_between_cases_reports_stopped():
    bid, _ids = _saved_run([("Q1", "k", "a"), ("Q2", "k", "a"), ("Q3", "k", "a")])
    n = {"calls": 0}

    def chat(_m):
        n["calls"] += 1
        return _reply("pass")

    summary = judge_benchmark(bid, _loaded(chat), should_stop=lambda: n["calls"] >= 1)
    assert summary.stopped and summary.judged == 1


# ── human review ─────────────────────────────────────────────────────────────


def test_human_verdict_beats_ai_and_can_be_retracted():
    _bid, ids = _saved_run([("Q1", "k", "a")])
    jdb.add_judgement(ids[0], kind="ai", verdict="fail", judge_model="m")
    jdb.add_judgement(ids[0], kind="human", verdict="pass", reasoning="synonym")
    assert db.get_case(ids[0])["verdict"] == "pass" and db.get_case(ids[0])["judge"] == "human"
    jdb.add_judgement(ids[0], kind="human", verdict="", reasoning="retracted")
    case = db.get_case(ids[0])
    assert case["verdict"] == "fail" and case["judge"] == "ai"  # back to the AI opinion


def test_agreement_between_ai_judges_and_the_human():
    bid, ids = _saved_run([("Q1", "k", "a"), ("Q2", "k", "a"), ("Q3", "k", "a")])
    for cid, ai, human in ((ids[0], "pass", "pass"), (ids[1], "pass", "fail"), (ids[2], "fail", None)):
        jdb.add_judgement(cid, kind="ai", verdict=ai, judge_model="gemma")
        if human:
            jdb.add_judgement(cid, kind="human", verdict=human)
    stats = jdb.agreement(bid)
    assert stats["human_reviewed"] == 2 and stats["judges"]["gemma"]["agreement"] == 0.5


def test_rescore_keeps_non_score_metadata():
    bid, ids = _saved_run([("Q1", "k", "a")])
    db.update_benchmark_scores(bid, {"eval_kind": "heldout", "retrieval": {"recall_at_k": 0.9}})
    jdb.add_judgement(ids[0], kind="human", verdict="pass")
    scores = rescore_benchmark(bid)
    assert scores["eval_kind"] == "heldout" and scores["retrieval"] == {"recall_at_k": 0.9} and scores["passed"] == 1


def test_there_is_no_scripted_judge_left_on_the_custom_dataset_path():
    for name in ("judge_case_heuristic", "judge_case_ai", "judge_case_local", "is_abstention", "_key_words"):
        assert not hasattr(judge_mod, name)
    import finetune_studio.testing.suite as suite_mod
    for name in ("apply_heuristic_judging", "_judge_each", "ensure_reasoning"):
        assert not hasattr(suite_mod, name)


# ── checklist (rubric) judging ───────────────────────────────────────────────


def _facts_reply(facts: list[tuple[str, str, str]], *, verdict: str = "pass", extra_wrong: bool = False) -> str:
    return json.dumps({
        "facts": [{"fact": f, "status": s, "evidence": e} for f, s, e in facts],
        "extra_wrong": extra_wrong, "reasoning": "checked", "verdict": verdict, "confidence": 0.9,
    })


_CASE = JudgeCase("What are the port and the user?", "2222; kcops", "Use SSH port 2222 and the **hubctl** user.", ["2222", "kcops"])


def test_checklist_decides_the_verdict_not_the_judges_own_summary():
    # the judge says "pass" but its own checklist shows kcops missing: the checklist wins, and says so
    res = parse_judge_reply(_facts_reply([("port 2222", "present", "port 2222"), ("user kcops", "missing", "")], verdict="pass"), _CASE)
    assert res.verdict == "partial"
    assert "the judge first said pass" in res.reasoning and "✔ port 2222" in res.reasoning and "✘ user kcops (missing)" in res.reasoning
    assert [f["status"] for f in res.facts] == ["present", "missing"]


def test_checklist_all_present_is_pass_none_present_is_fail():
    ok = parse_judge_reply(_facts_reply([("port", "present", "SSH port 2222"), ("user", "present", "hubctl")], verdict="fail"), _CASE)
    assert ok.verdict == "pass"
    none = parse_judge_reply(_facts_reply([("port", "wrong", "2220"), ("user", "missing", "")], verdict="partial"), _CASE)
    assert none.verdict == "fail"


def test_a_present_claim_with_a_quote_that_is_not_in_the_answer_is_not_trusted():
    res = parse_judge_reply(_facts_reply([("port", "present", "SSH port 2222"), ("user", "present", "the kcops account")]), _CASE)
    assert res.verdict == "partial"
    assert "? user — the judge's quote" in res.reasoning
    assert [f["status"] for f in res.facts] == ["present", "unverified"]


def test_quote_checking_ignores_case_spacing_and_markdown():
    case = JudgeCase("Rate?", "44.2", "The **44.2**  EUR/day  rate", ["44.2"])
    res = parse_judge_reply(_facts_reply([("rate", "present", "44.2 EUR/day")]), case)
    assert res.verdict == "pass"


def test_extra_wrong_claim_downgrades_an_otherwise_complete_answer():
    res = parse_judge_reply(_facts_reply([("port", "present", "port 2222"), ("user", "present", "hubctl")], extra_wrong=True), _CASE)
    assert res.verdict == "partial" and "adds a claim that contradicts the key" in res.reasoning


def test_unanswerable_cases_ignore_a_checklist_and_use_the_judges_verdict():
    case = JudgeCase("CEO's colour?", "", "I don't know.", expect_abstain=True)
    res = parse_judge_reply(_facts_reply([("n/a", "missing", "")], verdict="pass"), case)
    assert res.verdict == "pass"


def test_a_checklist_without_a_verdict_is_still_readable_but_not_without_a_case():
    reply = _facts_reply([("port", "present", "port 2222")])
    reply = json.dumps({k: v for k, v in json.loads(reply).items() if k != "verdict"})
    assert parse_judge_reply(reply, _CASE).verdict == "pass"            # the one listed fact is present and verified
    assert parse_judge_reply(reply).verdict == ""                       # no case to verify against and no verdict: unreadable
