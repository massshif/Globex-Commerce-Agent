# -*- coding: utf-8 -*-
"""TradeAgent

下单交易专家。基于 AgentScope 2.0 Agent，工具集：
create_order_tool / query_order_tool / cancel_order_tool。

同 SearchAgent 一样，通过 task_dispatch 工具被 MainAgent 调度，每次调度新建独立实例；
`build_tools()` 同时供 MainAgent 复用——主 Agent 持有同一批业务工具，可以不派发自己单干。
"""
from __future__ import annotations

from agentscope.agent import Agent, ReActConfig
from agentscope.tool import FunctionTool, Toolkit

from app.application.agents.context_policy import build_context_config
from app.application.agents.permissions import allow_business_tools
from app.application.prompts.loader import load_prompts
from app.application.tools.order_tools import (
    build_cancel_order_tool,
    build_create_order_tool,
    build_query_order_tool,
    build_show_shopping_form_tool,
)
from app.application.usecases.order_usecases import (
    CancelOrderUseCase,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from app.infrastructure.context_evidence import ContextEvidenceStore, EvidenceToolMiddleware
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.harness_middleware import HarnessToolMiddleware
from app.infrastructure.llm import create_chat_model
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)
from app.infrastructure.settings import Settings
from app.infrastructure.sql.trade_store import TradeStore
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.tracing import build_agent_middlewares


class TradeAgentFactory:
    def __init__(
        self,
        settings: Settings,
        place_order: PlaceOrderUseCase,
        query_order: QueryOrderUseCase,
        cancel_order: CancelOrderUseCase,
        bus: TradeEventBus,
        circuit_registry: CircuitBreakerRegistry,
        throttle: GatewayThrottle,
        sequencing=None,
        loop_detector=None,
        evidence_store: ContextEvidenceStore | None = None,
        trade_store: TradeStore | None = None,
    ) -> None:
        self._settings = settings
        self._place_order = place_order
        self._query_order = query_order
        self._cancel_order = cancel_order
        self._bus = bus
        self._circuit_registry = circuit_registry
        self._throttle = throttle
        self._sequencing = sequencing
        self._loop_detector = loop_detector
        self._evidence_store = evidence_store
        self._trade_store = trade_store

    def _resilience(self) -> list:
        chain = []
        if self._evidence_store is not None:
            chain.append(EvidenceToolMiddleware(self._evidence_store))
        if self._settings.harness_enabled and self._sequencing is not None and self._loop_detector is not None:
            chain.append(HarnessToolMiddleware(
                sequencing=self._sequencing, loop_detector=self._loop_detector, bus=self._bus,
            ))
        chain.append(ToolResilienceMiddleware(self._circuit_registry, self._bus))
        return chain

    def build_tools(self) -> list[FunctionTool]:
        """TradeAgent 的业务工具集，MainAgent 单干时持有同一批（均带超时+熔断保护）。"""
        return [
            FunctionTool(build_show_shopping_form_tool(self._bus), is_read_only=True, middlewares=self._resilience()),
            FunctionTool(
                build_create_order_tool(self._place_order, self._bus, self._trade_store),
                middlewares=self._resilience(),
            ),
            FunctionTool(
                build_query_order_tool(self._query_order, self._bus, self._trade_store),
                is_read_only=True,
                middlewares=self._resilience(),
            ),
            FunctionTool(
                build_cancel_order_tool(self._cancel_order, self._bus, self._trade_store),
                middlewares=self._resilience(),
            ),
        ]

    def build(self) -> Agent:
        prompts = load_prompts()["sub_agents"]["trade"]
        return allow_business_tools(
            Agent(
                name=prompts["name"],
                system_prompt=prompts["system_prompt"],
                model=create_chat_model(self._settings, throttle=self._throttle, bus=self._bus),
                toolkit=Toolkit(tools=list(self.build_tools())),
                middlewares=build_agent_middlewares(self._settings, self._evidence_store),
                context_config=build_context_config(
                    self._settings.context_size,
                    self._settings.tool_result_limit,
                ),
                react_config=ReActConfig(max_iters=6),
            ),
        )
