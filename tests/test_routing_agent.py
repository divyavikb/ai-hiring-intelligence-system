from src.agents.routing_agent import route_model, SMALL_MODEL, LARGE_MODEL


def test_simple_agent_gets_small_model():
    assert route_model("offer_insights", {"a": 1}).model == SMALL_MODEL


def test_complex_agents_get_large_model():
    for agent in ("sourcing_quality", "rejection_patterns", "panel_load", "pipeline_health"):
        assert route_model(agent, {"a": 1}).model == LARGE_MODEL


def test_force_model_overrides():
    assert route_model("offer_insights", {}, force_model=LARGE_MODEL).model == LARGE_MODEL