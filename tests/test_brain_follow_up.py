"""Follow-up search for questions that reach their answer through another item.

Quillmoor S, 376 judged answers, API answer model, default package (tests/corpus/quillmoor/RESULTS.md §6): the
first search finds the asset's row (it names the site) but never the site's profile (the question never names the
site), so two-step questions scored 10 %. Letting the model ask for one more search lifted them to 68 % and the
whole set from 81.1 % to 90.3 % (96.1 % with the bge reranker) with no loss on unanswerable questions. These tests pin the loop both the package's
/ask and the studio's Smart Brain chat run.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finetune_studio.smart_brain import standalone_server as srv


def _hit(cid: str, text: str, rank: int = 1) -> dict:
    return {"rank": rank, "chunk_id": cid, "filename": f"{cid}.txt", "text": text, "rrf_score": 0.03}


FIRST = [_hit("asset", "UVR-4655-R | UV reactor | Quellvoe Desalination Plant")]
SECOND = [_hit("site", "Quellvoe Desalination Plant is managed by Oona Halwicks."), FIRST[0]]


@pytest.mark.parametrize("reply, query", [
    ("SEARCH: Quellvoe Desalination Plant manager", "Quellvoe Desalination Plant manager"),
    ("**SEARCH:** site manager of Vellby Reservoir", "site manager of Vellby Reservoir"),
    ('search: "Dunells payment terms"', "Dunells payment terms"),
    # explanation first, the search on its own last line or after a sentence (how models actually write it)
    ("X is at Vellby Reservoir [a.tsv], but the context does not name its manager.\n\nSEARCH: Vellby Reservoir manager",
     "Vellby Reservoir manager"),
    ("The supplier is Kellwick [a.tsv]. SEARCH: Kellwick Marine Oy payment terms", "Kellwick Marine Oy payment terms"),
    ("The answer is Oona Halwicks [site.txt].", ""),
    ("Use the search box on the page.", ""),
    ("I think SEARCH: is a keyword", ""),
])
def test_follow_up_query_reads_a_search_line(reply: str, query: str) -> None:
    assert srv.follow_up_query(reply) == query


def test_merge_interleaves_new_passages_and_drops_repeats() -> None:
    hits = [_hit("a", "1"), _hit("b", "2", 2), _hit("c", "3", 3)]
    merged = srv.merge_follow_up_hits(hits, [_hit("b", "2"), _hit("x", "9"), _hit("y", "8")])
    assert [h["chunk_id"] for h in merged] == ["a", "x", "b", "y", "c"]   # a cut context keeps both searches' best
    assert [h["rank"] for h in merged] == [1, 2, 3, 4, 5]


def test_a_direct_answer_costs_one_call_and_no_search() -> None:
    chat = MagicMock(return_value="UVR-4655-R is at Quellvoe [asset.txt].")
    search = MagicMock()
    out = srv.answer_with_follow_ups("Where is UVR-4655-R?", FIRST, search, chat)
    assert out == {"answer": "UVR-4655-R is at Quellvoe [asset.txt].", "hits": FIRST, "follow_up_searches": []}
    search.assert_not_called()
    system = chat.call_args.args[0][0]["content"]
    assert system.startswith(srv.DEFAULT_SYSTEM_PROMPT) and srv.FOLLOW_UP_INSTRUCTION in system


def test_a_requested_search_is_run_and_the_last_round_cannot_search_again() -> None:
    replies = iter(["SEARCH: Quellvoe Desalination Plant manager", "Oona Halwicks [site.txt]."])
    seen: list[list[dict]] = []

    def chat(messages: list[dict]) -> str:
        seen.append(messages)
        return next(replies)

    search = MagicMock(return_value=SECOND)
    out = srv.answer_with_follow_ups("Who manages the site where UVR-4655-R is installed?", FIRST, search, chat)
    search.assert_called_once_with("Quellvoe Desalination Plant manager")
    assert out["answer"] == "Oona Halwicks [site.txt]."
    assert out["follow_up_searches"] == ["Quellvoe Desalination Plant manager"]
    assert [h["chunk_id"] for h in out["hits"]] == ["asset", "site"]
    assert srv.FOLLOW_UP_INSTRUCTION not in seen[1][0]["content"]          # the answer round offers no search
    assert "Oona Halwicks" in seen[1][0]["content"]                         # and carries the second hop


def test_a_search_asked_for_when_none_is_offered_never_reaches_the_user() -> None:
    chat = MagicMock(side_effect=["SEARCH: site manager", "SEARCH: site manager again"])
    out = srv.answer_with_follow_ups("q?", FIRST, MagicMock(return_value=SECOND), chat)
    assert out["answer"] == "I could not find the answer in the documents."


def test_a_repeated_search_is_not_run_twice_and_ends_with_an_answer() -> None:
    chat = MagicMock(side_effect=["SEARCH: Quellvoe manager", "SEARCH: quellvoe manager", "Oona Halwicks."])
    search = MagicMock(return_value=SECOND)
    out = srv.answer_with_follow_ups("q?", FIRST, search, chat, follow_ups=2)
    assert search.call_count == 1 and out["answer"] == "Oona Halwicks." and chat.call_count == 3


def test_follow_up_zero_is_the_old_single_answer() -> None:
    chat = MagicMock(return_value="SEARCH: anything")
    out = srv.answer_with_follow_ups("q?", FIRST, MagicMock(), chat, follow_ups=0)
    assert srv.FOLLOW_UP_INSTRUCTION not in chat.call_args.args[0][0]["content"]
    assert out["follow_up_searches"] == [] and chat.call_count == 1


def test_history_goes_between_the_context_and_the_question() -> None:
    chat = MagicMock(return_value="ok")
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    srv.answer_with_follow_ups("q?", FIRST, MagicMock(), chat, history=history, system_prompt="Be terse.")
    msgs = chat.call_args.args[0]
    assert msgs[0]["content"].startswith("Be terse. ") and msgs[1:] == [*history, {"role": "user", "content": "q?"}]


def test_follow_up_config_key_flag_env_and_range() -> None:
    cfg, origin = srv.resolve_config({}, {}, {})
    assert cfg["follow_up"] == 1 and origin["follow_up"] == "default"
    cfg, _ = srv.resolve_config({}, {"RAG_FOLLOW_UP": "0"}, {})
    assert cfg["follow_up"] == 0
    with pytest.raises(srv.ConfigError, match="follow_up"):
        srv.resolve_config({}, {"RAG_FOLLOW_UP": "9"}, {})


def test_package_answer_runs_the_loop_and_reports_the_search() -> None:
    corpus = MagicMock(spec=srv.Corpus)
    corpus.cfg = {**srv.DEFAULT_CONFIG, "answer": True, "llm_model": "m", "llm_base_url": "http://x/v1"}
    corpus.search.side_effect = lambda q, top_k=None: FIRST if q.startswith("Who") else SECOND
    corpus._chat.side_effect = ["SEARCH: Quellvoe Desalination Plant manager", "Oona Halwicks [site.txt]."]
    out = srv.Corpus.answer(corpus, "Who manages the site where UVR-4655-R is installed?")
    assert out["answer"] == "Oona Halwicks [site.txt]."
    assert out["follow_up_searches"] == ["Quellvoe Desalination Plant manager"]
    assert [s["filename"] for s in out["sources"]] == ["asset.txt", "site.txt"]


def test_studio_chat_runs_the_same_loop(client) -> None:
    pid = client.post("/api/projects", json={"name": "follow-up chat"}).json()["id"]
    q = MagicMock()
    q.search.side_effect = lambda query, top_k=10: [{**h, "source": h["filename"]} for h in
                                                   (FIRST if query.startswith("Who") else SECOND)]
    rag = MagicMock()
    rag.exists.return_value = True
    rag.load.return_value = q
    replies = iter(["SEARCH: Quellvoe Desalination Plant manager", "Oona Halwicks [site.txt]."])
    from finetune_studio.webui import app as webapp

    with patch("finetune_studio.smart_brain.BrainIndex", return_value=rag), \
         patch.object(webapp.inference_engine, "model", object()), \
         patch.object(webapp.inference_engine, "generate", side_effect=lambda *_a, **_k: next(replies)):
        r = client.post(f"/api/projects/{pid}/brain/chat",
                        json={"messages": [{"role": "user", "content": "Who manages the site where UVR-4655-R is?"}]})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["reply"] == "Oona Halwicks [site.txt]."
    assert out["follow_up_searches"] == ["Quellvoe Desalination Plant manager"]
    assert [s["filename"] for s in out["sources"]] == ["asset.txt", "site.txt"]


def test_brain_page_shows_the_extra_search() -> None:
    from pathlib import Path

    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/brain.html").read_text()
    assert "d.follow_up_searches" in html and "Also searched: " in html


def test_cli_brain_ask_runs_the_same_loop(tmp_path, capsys) -> None:
    import argparse
    import json

    from finetune_studio.cli.commands import brain as cli

    model = tmp_path / "model.gguf"
    model.write_text("x")
    index = MagicMock()
    index.search.side_effect = lambda query, top_k=5: FIRST if query.startswith("Who") else SECOND
    engine = MagicMock()
    engine.generate.side_effect = ["SEARCH: Quellvoe Desalination Plant manager", "Oona Halwicks [site.txt]."]
    args = argparse.Namespace(model=str(model), question="Who manages the site where UVR-4655-R is installed?",
                              top_k=5, max_tokens=64, system_prompt="", follow_up=1, json=True)
    with patch.object(cli, "_loaded", return_value=index), \
         patch("finetune_studio.inference.get_engine", return_value=engine), \
         patch("finetune_studio.inference.describe_worker", return_value="test worker"):
        cli._ask(args)
    out = json.loads(capsys.readouterr().out)
    assert out["response"] == "Oona Halwicks [site.txt]."
    assert out["follow_up_searches"] == ["Quellvoe Desalination Plant manager"]
    assert [s["filename"] for s in out["sources"]] == ["asset.txt", "site.txt"]
    engine.unload.assert_called_once()
