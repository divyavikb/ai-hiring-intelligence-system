import pytest

from src.rag.retriever import (
    index_knowledge_base,
    load_knowledge_documents,
    retrieve_context,
)

KB_FILES = {
    "sourcing.md",
    "rejection.md",
    "offers.md",
    "pipeline.md",
    "panel_load.md",
    "responsible_use.md",
}

# (query, agent tags to filter on, files the results may come from)
AGENT_CASES = {
    "sourcing": (
        "source conversion rejection rate sample size",
        ["sourcing", "responsible_use"],
    ),
    "rejection": (
        "rejection reasons job description vs interview",
        ["rejection", "responsible_use"],
    ),
    "panel_load": (
        "interviewer capacity utilisation rebalancing",
        ["panel_load", "responsible_use"],
    ),
    "offers": (
        "offer decline reasons acceptance rate compensation",
        ["offers", "responsible_use"],
    ),
    "pipeline": (
        "stage SLA bottleneck conversion",
        ["pipeline", "responsible_use"],
    ),
}


def test_knowledge_base_loads_sections():
    documents = load_knowledge_documents()

    assert len(documents) >= 20
    assert {doc["source"] for doc in documents} == KB_FILES
    assert any(
        doc["section"] == "Minimum sample size"
        for doc in documents
    )


def test_every_section_is_tagged_with_its_agent():
    for doc in load_knowledge_documents():
        assert doc["agent"] == doc["source"].removesuffix(".md")


def test_indexing_is_repeatable():
    first_count = index_knowledge_base()
    second_count = index_knowledge_base()

    assert first_count >= 20
    assert second_count == first_count


def test_retrieval_returns_source_and_section():
    index_knowledge_base()

    results = retrieve_context(
        "How should I compare recruiting sources using offer outcomes?",
        n_results=3,
    )

    assert len(results) > 0

    for result in results:
        assert result["text"]
        assert result["source"] in KB_FILES
        assert result["section"]


def test_retrieval_rejects_empty_query():
    with pytest.raises(ValueError, match="Query must not be empty"):
        retrieve_context("   ")


def test_sourcing_query_retrieves_sourcing_guideline():
    index_knowledge_base()

    results = retrieve_context(
        "Compare recruiting sources using stage conversion "
        "and offer outcomes. Account for small sample sizes.",
        n_results=3,
    )

    assert "sourcing.md" in [result["source"] for result in results]


@pytest.mark.parametrize("agent", AGENT_CASES)
def test_agent_filter_restricts_results_to_its_own_files(agent):
    index_knowledge_base()
    query, allowed_tags = AGENT_CASES[agent]

    results = retrieve_context(query, n_results=3, agents=allowed_tags)

    assert len(results) > 0
    allowed_files = {f"{tag}.md" for tag in allowed_tags}
    assert {result["source"] for result in results} <= allowed_files
    assert any(result["source"] == f"{agent}.md" for result in results)
