import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import metergraph
from metergraph import _context


def test_context_nests_merges_and_restores_after_exception():
    metergraph.set_default_tags(service="support")

    try:
        with metergraph.context(session_id="outer", tags={"customer": "acme"}):
            assert _context.snapshot().session_id == "outer"
            assert _context.snapshot().tags == {
                "service": "support",
                "customer": "acme",
            }

            try:
                with metergraph.session("inner"):
                    with metergraph.tags(customer="globex", region="us"):
                        assert _context.snapshot().session_id == "inner"
                        assert _context.snapshot().tags == {
                            "service": "support",
                            "customer": "globex",
                            "region": "us",
                        }
                        raise RuntimeError("boom")
            except RuntimeError:
                pass

            assert _context.snapshot().session_id == "outer"
            assert _context.snapshot().tags == {
                "service": "support",
                "customer": "acme",
            }

        assert _context.snapshot().session_id is None
        assert _context.snapshot().tags == {"service": "support"}
    finally:
        metergraph.set_default_tags()


def test_overlapping_async_jobs_do_not_leak_context():
    async def job(session_id: str):
        with metergraph.context(user_id=session_id, session_id=session_id, metadata={"job": session_id}):
            await asyncio.sleep(0)
            return _context.snapshot()

    async def run():
        return await asyncio.gather(job("one"), job("two"))

    one, two = asyncio.run(run())
    assert (one.session_id, one.tags) == ("one", {"job": "one"})
    assert (two.session_id, two.tags) == ("two", {"job": "two"})
    assert (one.user_id, two.user_id) == ("one", "two")
    assert _context.snapshot().session_id is None
    assert _context.snapshot().tags == {}


def test_context_decorator_and_wrapped_executor_propagate_scope():
    @metergraph.context(user_id="person", session_id="decorated", metadata={"source": "decorator"})
    async def decorated():
        with ThreadPoolExecutor(max_workers=1) as raw:
            executor = metergraph.wrap_executor(raw)
            return await asyncio.get_running_loop().run_in_executor(
                executor, _context.snapshot
            )

    observed = asyncio.run(decorated())
    assert observed.session_id == "decorated"
    assert observed.user_id == "person"
    assert observed.tags == {"source": "decorator"}


def test_legacy_setters_only_update_an_active_scope(caplog):
    with caplog.at_level(logging.WARNING, logger="metergraph"):
        metergraph.set_session("ignored")
        metergraph.set_tags(ignored=True)

    assert _context.snapshot().session_id is None
    assert _context.snapshot().tags == {}

    assert "active Metergraph context" in caplog.text

    with metergraph.route("active"):
        metergraph.set_session("accepted")
        metergraph.set_tags(tier="pro")
        assert _context.snapshot().session_id == "accepted"
        assert _context.snapshot().tags == {"tier": "pro"}

    assert _context.snapshot().session_id is None
    assert _context.snapshot().tags == {}


def test_user_and_metadata_inherit_override_clear_and_restore():
    values = {"customer_id": "first", "priority": "metadata"}
    metergraph.set_default_tags(service="backend")
    try:
        with metergraph.context(user_id=" person ", session_id="session", tags={"priority": 7}, metadata=values):
            values["customer_id"] = "mutated"
            assert _context.snapshot().tags == {"service": "backend", "priority": "metadata", "customer_id": "first"}
            with metergraph.context(user_id=None, metadata={"customer_id": 9, "new": "yes"}):
                assert _context.snapshot().user_id == " person "
                assert _context.snapshot().tags["customer_id"] == "first"
                metergraph.set_user("child")
                metergraph.set_user(3)
                assert _context.snapshot().user_id == "child"
                metergraph.set_user()
                metergraph.set_session(None)
                assert _context.snapshot().user_id is None
                assert _context.snapshot().session_id is None
            assert _context.snapshot().user_id == " person "
            assert _context.snapshot().session_id == "session"
        metergraph.set_user("outside")
        assert _context.snapshot().user_id is None
    finally:
        metergraph.set_default_tags()


def test_attribution_bounds_preserve_exact_strings_and_codepoint_order():
    values = {f"a{i:02}": "v" for i in range(31)} | {"😀": "astral", "～": "bmp", "api_key": "secret"}
    with metergraph.context(user_id="🙂" * 512, metadata=values):
        state = _context.snapshot()
        assert state.user_id == "🙂" * 512
        assert len(state.tags) == 32 and "～" in state.tags and "😀" not in state.tags
        assert "api_key" not in state.tags
        with metergraph.context(user_id="\x00", metadata={"bad": "\ud800", "nested": {}, "ok": ""}):
            assert _context.snapshot().user_id == state.user_id
            assert "bad" not in _context.snapshot().tags and "nested" not in _context.snapshot().tags
    with metergraph.context(user_id="\u0085", metadata={f"k{i:02}": "🙂" * 1000 for i in range(32)}):
        import json
        tags = _context.snapshot().tags
        assert _context.snapshot().user_id is None
        assert 0 < len(tags) < 32
        assert len(json.dumps(tags, ensure_ascii=False, separators=(",", ":")).encode()) <= 65536
