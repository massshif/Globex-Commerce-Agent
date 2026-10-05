from __future__ import annotations

import pytest

from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository
from app.infrastructure.sql.trade_store import TradeConflict, TradeStore

ADDRESS = {
    "recipient_name": "张三", "country": "CN", "state": "浙江", "city": "杭州",
    "address_line": "西湖区 1 号", "postal_code": "310000", "phone": "13800000000",
}


@pytest.mark.asyncio
async def test_trade_store_uses_injected_catalog_and_is_idempotent(tmp_path):
    products = await InMemoryProductRepository().list_all()
    store = TradeStore(tmp_path / "trade.db", products)
    draft = store.prepare_order("buyer", "session", [{
        "product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1,
    }], ADDRESS)
    result = store.decide(draft["operation_id"], "buyer", "session", draft["draft_hash"], True)
    assert result["status"] == "confirmed"
    assert store.decide(draft["operation_id"], "buyer", "session", draft["draft_hash"], True) == result


@pytest.mark.asyncio
async def test_trade_store_rejects_stale_or_cross_buyer_approval(tmp_path):
    products = await InMemoryProductRepository().list_all()
    store = TradeStore(tmp_path / "trade.db", products)
    draft = store.prepare_order("buyer", "session", [{
        "product_id": "P1008", "sku_id": "P1008-S1", "quantity": 1,
    }], ADDRESS)
    with pytest.raises(TradeConflict):
        store.decide(draft["operation_id"], "other", "session", draft["draft_hash"], True)
    with pytest.raises(TradeConflict):
        store.decide(draft["operation_id"], "buyer", "session", "stale", True)
