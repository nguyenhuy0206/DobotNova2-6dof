"""Explore mode: Astra learns the cell by itself from primitive tools (scripts/run_explore.sh).

    config        config/explore.yaml settings
    prompt        the system prompt (task-neutral)
    tools         tools only explore has: knowledge, history, routines, skill versions, view guard
    diagnosis     failures -> hypotheses with tests -> conclusions, linked to the knowledge store
    learning      the review after each episode: outcome, skills, knowledge
    session       ExploreSession (preflight, run_episode, reflect) and build_explore

    session = build_explore()
    problems = session.preflight()
    session.run_episode(session.cfg.task)
"""
from agent.explore.config import ExploreConfig, load_explore_config
from agent.explore.prompt import EXPLORE_INSTRUCTIONS
from agent.explore.session import ExploreSession, build_explore, compact_state
from agent.explore.tools import code_specs, explore_specs, history_spec, knowledge_specs

__all__ = ["EXPLORE_INSTRUCTIONS", "ExploreConfig", "ExploreSession", "build_explore", "code_specs",
           "compact_state", "explore_specs", "history_spec", "knowledge_specs", "load_explore_config"]
