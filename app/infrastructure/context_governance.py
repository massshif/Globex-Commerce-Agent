"""Archive tool evidence before AgentScope may summarize the conversation."""
from __future__ import annotations

import logging
from copy import deepcopy

from agentscope.message import AssistantMsg, TextBlock, ToolCallBlock, ToolResultBlock
from agentscope.middleware import MiddlewareBase

from app.infrastructure.context_evidence import ContextEvidenceStore

logger = logging.getLogger(__name__)


def _pairs(agent):
    pairs: list[tuple[ToolResultBlock, dict]] = []
    calls: dict[str, dict] = {}
    for message in agent.state.context:
        for block in message.get_content_blocks():
            if isinstance(block, ToolCallBlock):
                calls[block.id] = {"name": block.name, "input": block.input,
                                   "block": block, "speaker": message.name}
            elif isinstance(block, ToolResultBlock):
                pairs.append((block, calls.get(block.id, {"name": block.name, "input": ""})))
    return pairs


def archive_results(agent, store: ContextEvidenceStore, buyer_id: str, session_id: str) -> None:
    """Cover restored sessions and framework tools that bypass tool middleware."""
    for block, call in _pairs(agent):
        if block.metadata.get("evidence_id"):
            continue
        text = block.output if isinstance(block.output, str) else "\n".join(
            str(getattr(item, "text", "")) for item in block.output
        )
        if text and not text.startswith("[evidence:"):
            block.metadata["evidence_id"] = store.save(
                buyer_id, session_id, call["name"], {"input": call["input"]}, text,
            )


def prune_read_results(agent, store: ContextEvidenceStore, buyer_id: str,
                       session_id: str, keep_recent: int = 5) -> int:
    archive_results(agent, store, buyer_id, session_id)
    pairs = _pairs(agent)
    changed = 0
    old_pairs = pairs if keep_recent <= 0 else pairs[:-keep_recent]
    for block, call in old_pairs:
        if block.metadata.get("evidence_compacted"):
            continue
        text = block.output if isinstance(block.output, str) else "\n".join(
            str(getattr(item, "text", "")) for item in block.output
        )
        if not text or text.startswith("[evidence:"):
            continue
        evidence_id = block.metadata["evidence_id"]
        block.output = [TextBlock(text=f"[evidence:{evidence_id}] 已读工具结果已归档；需要细节请按证据编号回查。")]
        block.metadata["evidence_id"] = evidence_id
        block.metadata["evidence_compacted"] = True
        changed += 1
    return changed


class EvidenceCompressionMiddleware(MiddlewareBase):
    """Keep the latest complete tool pairs across AgentScope summarization.

    The framework reserves a token suffix, not a count of tool results. Its
    suffix may contain fewer than five results, so restore missing pairs after
    compression. Complete source records are saved before calling the framework.
    """

    def __init__(self, store: ContextEvidenceStore, keep_recent: int = 5) -> None:
        self.store = store
        self.keep_recent = max(0, keep_recent)

    async def on_compress_context(self, agent, input_kwargs, next_handler) -> None:
        from app.infrastructure.context import ShoppingContext

        snapshot = ShoppingContext.current()
        if snapshot is None:
            await next_handler(**input_kwargs)
            return
        buyer_id, session_id = snapshot.buyer_id, snapshot.shopping_session_id
        prune_read_results(agent, self.store, buyer_id, session_id, self.keep_recent)
        protected = []
        for result, call in (_pairs(agent)[-self.keep_recent:] if self.keep_recent else []):
            if call.get("block") is not None:
                protected.append((result.id, AssistantMsg(
                    call["speaker"], [deepcopy(call["block"]), deepcopy(result)],
                )))
        summary_before = agent.state.summary
        await next_handler(**input_kwargs)
        if agent.state.summary == summary_before:
            return
        present = {result.id for result, _ in _pairs(agent)}
        missing = [message for result_id, message in protected if result_id not in present]
        if missing:
            agent.state.context = missing + agent.state.context
            await self._fit_restored_context(agent, missing)

    async def _fit_restored_context(self, agent, restored: list[tuple[str, AssistantMsg]]) -> None:
        """Restore as many recent pairs as fit, never exceed the model window."""
        try:
            model_limit = int(agent.model.context_size)
            trigger = float(getattr(agent.context_config, "trigger_ratio", 0.75))

            async def count() -> int:
                kwargs = await agent._prepare_model_input()  # AgentScope's canonical counter input
                return int(await agent.model.count_tokens(**kwargs))

            # Keep headroom so the next user message does not immediately overflow.
            while restored and await count() >= int(model_limit * trigger):
                _, message = restored.pop(0)
                for block in message.get_content_blocks():
                    if isinstance(block, ToolResultBlock):
                        evidence_id = block.metadata.get("evidence_id")
                        if evidence_id:
                            block.output = [TextBlock(
                                text=f"[evidence:{evidence_id}] 已读工具结果已归档；需要细节请按证据编号回查。",
                            )]
                            block.metadata["evidence_compacted"] = True
                # The message remains in context as a compact evidence reference.
            if await count() >= model_limit:
                logger.warning("摘要与受保护工具结果仍接近模型上下文上限，继续依赖 AgentScope 下一轮压缩")
        except Exception as err:  # noqa: BLE001
            logger.warning("恢复工具结果后的 token 校验跳过：%s", err)
