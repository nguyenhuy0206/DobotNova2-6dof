"""Qwen's robot runs are marked and kept out of what Astra reads."""
from memory.store import EpisodeStore, driver_of


def test_driver_of_meta():
    assert driver_of({"model": "gpt-6-astra"}) == "astra"
    assert driver_of({"model": "qwen-local:qwen3.5-4b-4bit-frozen"}) == "qwen"     # before meta.driver
    assert driver_of({"model": "x", "driver": "qwen"}) == "qwen"
    assert driver_of(None) == "astra"


def test_episode_list_is_astra_only_by_default(tmp_path):
    st = EpisodeStore(tmp_path)
    a = st.start("task", {"model": "gpt-6-astra", "driver": "astra"})
    q = st.start("task", {"model": "qwen-local:m"})
    assert st.list() == [a]
    assert st.list(driver="qwen") == [q]
    assert sorted(st.list(driver=None)) == sorted([a, q])


def test_qwen_session_writes_nothing_of_astras(monkeypatch):
    from agent.explore import build_explore
    from agent.explore.session import QWEN_NO_WRITE
    from common.paths import STUDENT_RUNS_DIR
    s = build_explore(mode="dry_run", client=object(), driver="qwen")
    assert not set(QWEN_NO_WRITE) & set(s.executor.specs)
    assert s.knowledge.obs.path == STUDENT_RUNS_DIR / "observations.jsonl"
    a = build_explore(mode="dry_run", client=object(), driver="astra")
    assert "save_skill" in a.executor.specs and a.knowledge.obs.path.parent.name == "memory"


def test_qwen_episodes_are_never_reviewed_into_astras_memory(tmp_path):
    import pytest

    from agent.explore import build_explore
    s = build_explore(mode="dry_run", client=object(), driver="qwen")
    s.harness.episodes = EpisodeStore(tmp_path)
    ep = s.harness.episodes.start("task", {"model": "qwen-local:m", "driver": "qwen"})
    with pytest.raises(ValueError, match="not run by Astra"):
        s.reflect(ep)


def test_the_review_harness_runs_no_judge(tmp_path):
    from agent.harness import AgentHarness
    from common.config import AgentConfig
    h = AgentHarness(None, None, AgentConfig(), episodes=EpisodeStore(tmp_path), judge=False)
    assert h.judge is None
