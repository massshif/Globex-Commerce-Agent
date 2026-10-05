import tempfile
from pathlib import Path

import pytest
from agentscope.message import AssistantMsg, ToolCallBlock, ToolResultBlock
from agentscope.state import AgentState

from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_evidence import ContextEvidenceStore
from app.infrastructure.context_governance import EvidenceCompressionMiddleware, prune_read_results


def test_evidence_is_buyer_and_session_scoped():
    store = ContextEvidenceStore(Path(tempfile.mkdtemp()))
    evidence_id = store.save("buyer-a", "session-a", "product_search_tool", {"q": "bag"}, "full result")
    assert store.get("buyer-a", "session-a", evidence_id)["result"] == "full result"
    assert store.get("buyer-b", "session-a", evidence_id) is None
    assert store.get("buyer-a", "session-b", evidence_id) is None


def test_prune_replaces_old_result_with_reference():
    call = ToolCallBlock(id="call-1", name="product_search_tool", input="{}")
    result = ToolResultBlock(id="call-1", name="product_search_tool", output="full result")
    agent = type("AgentStub", (), {})()
    agent.state = AgentState(context=[AssistantMsg("agent", [call, result])])
    store = ContextEvidenceStore(Path(tempfile.mkdtemp()))

    changed = prune_read_results(agent, store, "buyer-a", "session-a", keep_recent=0)

    assert changed == 1
    compacted = agent.state.context[0].content[-1]
    assert "[evidence:" in compacted.output[0].text
    assert len(store.list_recent("buyer-a", "session-a")) == 1


@pytest.mark.asyncio
async def test_compression_middleware_restores_latest_tool_pair():
    call = ToolCallBlock(id="call-1", name="product_search_tool", input="{}")
    result = ToolResultBlock(id="call-1", name="product_search_tool", output="full result")
    agent = type("AgentStub", (), {})()
    agent.state = AgentState(context=[AssistantMsg("agent", [call, result])])
    agent.state.summary = None
    store = ContextEvidenceStore(Path(tempfile.mkdtemp()))
    middleware = EvidenceCompressionMiddleware(store, keep_recent=5)
    token = ShoppingContext.set(ShoppingContextSnapshot("session-a", "buyer-a", "en-US", "USD"))
    try:
        async def compress_everything(**_kwargs):
            agent.state.summary = "summary"
            agent.state.context = []

        await middleware.on_compress_context(agent, {}, compress_everything)
    finally:
        ShoppingContext.reset(token)

    assert len(agent.state.context) == 1
    assert agent.state.context[0].content[-1].output == "full result"
    assert len(store.list_recent("buyer-a", "session-a")) == 1
