# -*- coding: utf-8 -*-
"""FastAPI 服务入口

路由：
    POST /commerce/intents                 提交买家意图（同步返回最终回复；启用队列时内部入队后等结果）
    POST /commerce/intents/async           提交买家意图（立即返回 task_id，结果走 WS 或轮询）
    GET  /commerce/tasks/{task_id}         查任务状态（queued / running / done / failed）
    WS   /commerce/events                  订阅会话事件流
    GET  /commerce/orders/{order_id}       查询订单（直连 UseCase，不过 Agent）
    POST /commerce/orders/{order_id}/cancel  取消订单（直连 UseCase）
    GET  /health                           健康检查（含依赖连通性与队列深度）

启动：
    uv run uvicorn app.presentation.server:app --port 8000
    uv run python -m app.worker          # 启用队列时另起消费进程

同步接口为什么保留：13 case 评测脚本与前端都依赖它直接返回 final_text，
改成纯异步会一次性搞挂回归与前端。削峰由 worker 并发度保证，与接口形态无关。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.application.agents.orchestrator import SubmitIntentInput
from app.composition import Container, build_container
from app.domain.queue.ports.task_queue import IntentTask, TaskStatus
from app.presentation.connection import ConnectionManager
from app.presentation.dto import (
    CancelOrderRequest,
    PermissionConfirmRequest,
    SubmitIntentRequest,
    SubmitIntentResponse,
    TradeConfirmRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

logger = logging.getLogger(__name__)

# 幂等键有效期：同一会话同一句话在此窗口内重复提交视为重复请求
_IDEMPOTENCY_TTL_SECONDS = 600
# 轮数计数器存活时长：比幂等窗口长得多，让一整段会话都能被正确分类
_TURN_COUNTER_TTL_SECONDS = 86400


def build_app() -> FastAPI:
    state: dict = {}

    async def _forward_remote_events(c: Container) -> None:
        """把其他进程（worker）广播的事件转发给本进程的 WS 订阅者。"""
        if c.backplane is None:
            return
        try:
            async for event in c.backplane.listen():
                c.bus.deliver_local(event)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            logger.warning("事件背板监听中断：%s", err)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        c = await build_container()
        state["c"] = c
        state["connections"] = ConnectionManager(c.bus)
        await c.startup()
        if c.backplane is not None:
            # 跨进程事件转发：不开这个任务，worker 产生的流式事件到不了前端
            state["forwarder"] = asyncio.create_task(_forward_remote_events(c))
        try:
            yield
        finally:
            forwarder = state.pop("forwarder", None)
            if forwarder is not None:
                forwarder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder
            state.pop("c", None)
            await c.shutdown()

    api = FastAPI(title="Globex 跨境电商 Agent", version="0.4.0", lifespan=lifespan)

    def container() -> Container:
        if "c" not in state:
            raise HTTPException(status_code=503, detail="服务尚未就绪")
        return state["c"]

    settings_origins = build_container_origins()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health() -> dict:
        """依赖连通性一并报出，避免"进程活着但存储已挂"被当成健康。"""
        c = container()
        database = "disabled"
        if c.db_engine is not None:
            try:
                async with c.db_engine.connect() as conn:
                    await conn.execute(text("select 1"))
                database = c.db_engine.url.get_backend_name()
            except Exception as err:  # noqa: BLE001
                database = f"error: {err}"
        redis_state = "disabled"
        if c.cache.enabled:
            redis_state = "ok" if await c.cache.ping() else "error"
        return {
            "status": "ok",
            "model": c.settings.llm_model,
            "database": database,
            "redis": redis_state,
            "semantic_cache": c.semantic_cache.enabled,
            "queue": "enabled" if c.task_queue is not None else "disabled",
            "queue_depth": await c.task_queue.depth() if c.task_queue is not None else 0,
        }

    @api.post("/commerce/intents", response_model=SubmitIntentResponse)
    async def submit_intent(body: SubmitIntentRequest) -> SubmitIntentResponse:
        c = container()
        session_id = body.shopping_session_id or f"session-{uuid.uuid4().hex[:8]}"
        intent = SubmitIntentInput(
            shopping_session_id=session_id,
            buyer_id=body.buyer_id,
            locale=body.locale,
            currency=body.currency,
            raw_query=body.raw_query,
        )
        if c.task_queue is None:
            result = await c.orchestrator.handle_intent(intent)
            return SubmitIntentResponse(
                shopping_session_id=result.shopping_session_id, final_text=result.final_text,
            )

        task_id = await _enqueue(c, intent)
        final_text = await _await_result(c, task_id, session_id)
        return SubmitIntentResponse(shopping_session_id=session_id, final_text=final_text)

    @api.post("/ag-ui/agent")
    async def ag_ui_agent(body: dict) -> StreamingResponse:
        """AG-UI HTTP endpoint: accepts the official RunAgentInput shape.

        The legacy commerce request fields remain accepted for CLI compatibility.
        """
        c = container()
        messages = body.get("messages") or []
        last_message = messages[-1] if messages else {}
        content = last_message.get("content", "") if isinstance(last_message, dict) else ""
        # AG-UI uses camelCase field names; keep snake_case aliases for the
        # legacy CLI payloads accepted by this endpoint.
        forwarded_props = body.get("forwarded_props") or body.get("forwardedProps") or {}
        session_id = (
            body.get("thread_id")
            or body.get("threadId")
            or body.get("shopping_session_id")
            or f"session-{uuid.uuid4().hex[:8]}"
        )
        buyer_id = body.get("buyer_id") or forwarded_props.get("buyer_id", "browser")
        intent = SubmitIntentInput(
            shopping_session_id=session_id,
            buyer_id=buyer_id,
            locale=body.get("locale", "zh-CN"),
            currency=body.get("currency", "CNY"),
            raw_query=body.get("raw_query") or content,
        )
        queue = c.bus.subscribe(session_id)
        run_id = f"run-{uuid.uuid4().hex}"
        message_id = f"msg-{uuid.uuid4().hex}"

        async def stream():
            def frame(event: dict) -> str:
                return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            try:
                yield frame({"type": "RUN_STARTED", "threadId": session_id, "runId": run_id})
                yield frame({"type": "TEXT_MESSAGE_START", "messageId": message_id, "role": "assistant"})
                task = asyncio.create_task(_run_intent(c, intent))
                saw_tokens = False
                tool_ids: dict[str, str] = {}
                while True:
                    if task.done() and queue.empty():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if event.type == "token.delta":
                        saw_tokens = True
                    mapped = _to_ag_ui_events(event, session_id, run_id, message_id, tool_ids)
                    if event.type == "final.result" and saw_tokens:
                        mapped = [item for item in mapped if item.get("type") != "TEXT_MESSAGE_CONTENT"]
                    for item in mapped:
                        yield frame(item)
                    if event.type == "final.result":
                        yield frame({"type": "TEXT_MESSAGE_END", "messageId": message_id})
                        yield frame({"type": "RUN_FINISHED", "threadId": session_id, "runId": run_id})
                    elif event.type == "error":
                        yield frame({"type": "RUN_ERROR", "threadId": session_id, "runId": run_id,
                                     "message": str(event.payload.get("message", "error"))})
                await task
            except Exception as err:  # noqa: BLE001
                yield frame({"type": "RUN_ERROR", "threadId": session_id, "runId": run_id,
                             "message": str(err)})
            finally:
                c.bus.unsubscribe(session_id, queue)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @api.post("/commerce/intents/async")
    async def submit_intent_async(body: SubmitIntentRequest) -> dict:
        c = container()
        session_id = body.shopping_session_id or f"session-{uuid.uuid4().hex[:8]}"
        intent = SubmitIntentInput(
            shopping_session_id=session_id,
            buyer_id=body.buyer_id,
            locale=body.locale,
            currency=body.currency,
            raw_query=body.raw_query,
        )
        if c.task_queue is None:
            raise HTTPException(status_code=503, detail="队列未启用，请使用 /commerce/intents")
        task_id = await _enqueue(c, intent)
        return {"shopping_session_id": session_id, "task_id": task_id, "state": "queued"}

    @api.post("/commerce/sessions/{session_id}/confirm")
    async def confirm_session_permission(
        session_id: str, body: PermissionConfirmRequest,
    ) -> dict:
        result = await container().orchestrator.confirm_pending(session_id, body.confirmed)
        return {"shopping_session_id": session_id, "final_text": result}

    @api.post("/commerce/sessions/{session_id}/trade-confirm")
    async def confirm_trade(session_id: str, body: TradeConfirmRequest) -> dict:
        c = container()
        # buyer_id is part of the authenticated/request context in this MVP;
        # require the existing session's last intent to prevent cross-buyer approval.
        intent = c.orchestrator.last_intent(session_id)
        if intent is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            result = c.trade_store.decide(body.operation_id, intent.buyer_id, session_id,
                                          body.draft_hash, body.approved)
        except ValueError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        c.bus.publish(session_id, "confirmation.result", result)
        c.bus.publish(session_id, "final.result", {"text": json.dumps(result, ensure_ascii=False)})
        return result

    @api.get("/commerce/tasks/{task_id}")
    async def get_task(task_id: str) -> dict:
        c = container()
        if c.task_queue is None:
            raise HTTPException(status_code=503, detail="队列未启用")
        status = await c.task_queue.get_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务不存在或已过期：{task_id}")
        return {
            "task_id": status.task_id,
            "state": status.state,
            "final_text": status.final_text,
            "error": status.error,
            "queue_position": status.queue_position,
        }

    @api.websocket("/commerce/events")
    async def commerce_events(websocket: WebSocket) -> None:
        await state["connections"].serve(websocket)

    @api.get("/commerce/orders/{order_id}")
    async def get_order(order_id: str) -> dict:
        try:
            return await container().query_order.execute(order_id)
        except ValueError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @api.get("/commerce/preferences/{buyer_id}")
    async def list_preferences(buyer_id: str) -> list[dict]:
        return [
            {"buyer_id": item.buyer_id, "kind": item.kind, "statement": item.statement,
             "created_at": item.created_at}
            for item in await container().preference_store.list_by_buyer(buyer_id)
        ]

    @api.get("/commerce/sessions/{session_id}/messages")
    async def list_session_messages(session_id: str, buyer_id: str) -> list[dict]:
        """Restore the durable conversation transcript for one buyer session."""
        c = container()
        session = await c.conversation_store.find_session(session_id)
        if session is None or session["buyer_id"] != buyer_id:
            raise HTTPException(status_code=404, detail="会话不存在或不属于当前买家")
        turns = await c.conversation_store.list_turns(session_id)
        return [
            {
                "role": turn.role,
                "text": turn.content,
                "created_at": turn.created_at,
            }
            for turn in turns
        ]

    @api.get("/commerce/sessions")
    async def list_sessions(buyer_id: str, limit: int = 50) -> list[dict]:
        return await container().conversation_store.list_sessions(buyer_id, limit=min(limit, 100))

    @api.get("/commerce/sessions/{session_id}/events")
    async def list_session_events(session_id: str, buyer_id: str) -> list[dict]:
        """Restore the event timeline and product-search results for a session."""
        c = container()
        session = await c.conversation_store.find_session(session_id)
        if session is None or session["buyer_id"] != buyer_id:
            raise HTTPException(status_code=404, detail="会话不存在或不属于当前买家")
        return [
            {"type": event.type, "payload": event.payload, "occurred_at": event.occurred_at}
            for event in await c.conversation_store.list_events(session_id)
        ]

    @api.get("/commerce/context-evidence/{buyer_id}/{session_id}")
    async def list_context_evidence(
        buyer_id: str, session_id: str, page: int = 1, limit: int = 5,
    ) -> list[dict]:
        """Paginated lookup for archived tool evidence; buyer/session are part of the key."""
        return container().evidence_store.list_recent(buyer_id, session_id, page=page, limit=limit)

    @api.get("/commerce/sessions/{session_id}/history")
    async def session_history(session_id: str, buyer_id: str) -> list[dict]:
        """Reload durable business events after a page refresh; no model internals are replayed."""
        return container().trade_store.history(buyer_id, session_id)

    @api.get("/commerce/context-evidence/{buyer_id}/{session_id}/{evidence_id}")
    async def get_context_evidence(buyer_id: str, session_id: str, evidence_id: str) -> dict:
        evidence = container().evidence_store.get(buyer_id, session_id, evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="证据不存在或不属于当前买家会话")
        return evidence

    @api.get("/commerce/orders")
    async def list_orders(buyer_id: str) -> list[dict]:
        return container().trade_store.list_orders(buyer_id)

    @api.post("/commerce/orders/{order_id}/cancel")
    async def cancel_order_endpoint(order_id: str, body: CancelOrderRequest) -> dict:
        try:
            return await container().cancel_order.execute(order_id, body.reason)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    return api


async def _run_intent(c: Container, intent: SubmitIntentInput) -> str:
    """Run through the configured queue when present, otherwise in-process."""
    if c.task_queue is None:
        return (await c.orchestrator.handle_intent(intent)).final_text
    task_id = await _enqueue(c, intent)
    return await _await_result(c, task_id, intent.shopping_session_id)


def _to_ag_ui_events(event, session_id: str, run_id: str, message_id: str,
                     tool_ids: dict[str, str] | None = None) -> list[dict]:
    """Map a Globex event to the stable AG-UI event vocabulary."""
    payload = event.payload if isinstance(event.payload, dict) else {"value": event.payload}
    if event.type == "token.delta":
        return [{"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id,
                 "delta": str(payload.get("token", ""))}]
    if event.type == "tool.invoke":
        tool_name = str(payload.get("tool", "tool"))
        tool_id = f"tool-{uuid.uuid4().hex}"
        if tool_ids is not None:
            tool_ids[tool_name] = tool_id
        args = json.dumps(payload.get("args", {}), ensure_ascii=False)
        return [
            {"type": "TOOL_CALL_START", "toolCallId": tool_id,
             "toolCallName": payload.get("tool", "tool")},
            {"type": "TOOL_CALL_ARGS", "toolCallId": tool_id, "delta": args},
            {"type": "TOOL_CALL_END", "toolCallId": tool_id},
        ]
    if event.type == "tool.result":
        tool_name = str(payload.get("tool", "tool"))
        return [{"type": "TOOL_CALL_RESULT", "messageId": message_id,
                 "toolCallId": (tool_ids or {}).get(tool_name, payload.get("tool_call_id", "unknown")),
                 "content": json.dumps(payload, ensure_ascii=False)}]
    if event.type == "final.result":
        # Cached replies may not emit token.delta; emit content once in that case.
        return [{"type": "TEXT_MESSAGE_CONTENT", "messageId": message_id,
                 "delta": str(payload.get("text", ""))}]
    if event.type == "agent.dispatch":
        return [{"type": "STEP_STARTED", "stepName": payload.get("agent", "agent")}]
    if event.type == "plan.update":
        return [{"type": "STATE_SNAPSHOT", "snapshot": payload}]
    if event.type == "shopping.form":
        return [{"type": "CUSTOM", "name": "globex.shopping_form", "value": payload}]
    if event.type == "error":
        return []
    # Preserve domain-specific observability without pretending it is a core AG-UI event.
    return [{"type": "CUSTOM", "name": f"globex.{event.type}", "value": payload}]


async def _queue_priority(c: Container, session_id: str) -> int:
    """按对话轮数定队列优先级（0 = 正常，1 = 大请求）。

    长会话上下文大、单次耗时长，分到低优先流，避免堵住新会话。
    轮数计数存 Redis（计数不原子，但优先级本身是启发式，差一两次无影响）；
    Redis 不可用或开关关闭时一律返回 0，退化为单队列。
    """
    if not c.settings.queue_priority_enabled or not c.cache.enabled:
        return 0
    key = f"globex:turns:{session_id}"
    try:
        current = int(await c.cache.get_raw(key) or 0) + 1
        await c.cache.set_json(key, current, _TURN_COUNTER_TTL_SECONDS)
    except Exception:  # noqa: BLE001 —— 计数失败不能影响入队
        return 0
    return 1 if current >= c.settings.queue_large_request_turns else 0


async def _enqueue(c: Container, intent: SubmitIntentInput) -> str:
    """入队并做幂等保护。

    队列是 at-least-once，且买家/前端可能重复提交。用「会话 + 问句」指纹做幂等键，
    命中说明短时间内已提交过同样内容，直接复用原 task_id，不再入队一次。
    这一步对写操作（下单）尤其关键：重复消费等于重复下单。
    """
    fingerprint = hashlib.sha256(
        f"{intent.shopping_session_id}\n{intent.raw_query}".encode(),
    ).hexdigest()[:32]
    idem_key = f"idem:{fingerprint}"
    task_id = f"task-{uuid.uuid4().hex[:12]}"

    acquired = await c.cache.set_if_absent(idem_key, task_id, _IDEMPOTENCY_TTL_SECONDS)
    if not acquired:
        previous = await c.cache.get_raw(idem_key)
        if previous:
            logger.info("幂等命中，复用已有任务：%s", previous)
            return previous

    await c.task_queue.enqueue(  # type: ignore[union-attr]
        IntentTask(
            task_id=task_id,
            shopping_session_id=intent.shopping_session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
            raw_query=intent.raw_query,
            priority=await _queue_priority(c, intent.shopping_session_id),
        ),
    )
    await c.task_queue.set_status(TaskStatus(task_id=task_id, state="queued"))  # type: ignore[union-attr]
    c.bus.publish(intent.shopping_session_id, "task.queued", {"task_id": task_id})
    return task_id


async def _await_result(c: Container, task_id: str, session_id: str) -> str:
    """等 worker 跑完。

    优先等 final.result 事件（实时）；同时定期查任务状态兜底——
    worker 崩溃或任务进死信时事件永远不会来，只靠等事件会把请求挂死。
    """
    queue = c.bus.subscribe(session_id)
    deadline = time.monotonic() + c.settings.queue_wait_seconds
    try:
        while time.monotonic() < deadline:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                status = await c.task_queue.get_status(task_id)  # type: ignore[union-attr]
                if status is not None and status.state == "done":
                    return status.final_text
                if status is not None and status.state == "failed":
                    return f"[error] {status.error}"
                continue
            if event.type == "final.result":
                return str(event.payload.get("text", ""))
        return "[error] 处理超时，请稍后重试或改用异步接口查询任务状态"
    finally:
        c.bus.unsubscribe(session_id, queue)


def build_container_origins() -> list[str]:
    """CORS 需要在 app 构造期就确定，此处单独读一次配置。"""
    from app.infrastructure.settings import load_settings

    return load_settings().cors_origins


app = build_app()


if __name__ == "__main__":
    import uvicorn

    from app.infrastructure.settings import load_settings

    uvicorn.run(app, host="0.0.0.0", port=load_settings().port)
