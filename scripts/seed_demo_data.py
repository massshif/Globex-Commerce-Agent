"""Seed repeatable local data for the recording/demo environment.

The product catalog is intentionally an in-memory catalog and is rebuilt on
startup. This script seeds only sample confirmed/cancelled orders in the local
SQLite store. Personal preferences are never seeded: they must come from an
explicit, confirmed user interaction.

Usage::

    uv run python scripts/seed_demo_data.py
    uv run python scripts/seed_demo_data.py --buyer-id buyer-demo

The inserts are idempotent and use the ``demo-``/``GBX-DEMO-`` namespaces so
they can be identified and removed without touching real records.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "globex.db"

ORDER_TEMPLATES = [
    ("P1004", "P1004-S1", "AeroHush 主动降噪蓝牙耳机 Pro", 21900, "USD", 1, "CONFIRMED"),
    ("P1023", "P1023-S1", "SilentBuds 主动降噪耳塞式耳机", 18900, "USD", 1, "CONFIRMED"),
    ("P1001", "P1001-S1", "Nomadica 旅行三件套", 18900, "CNY", 1, "CONFIRMED"),
    ("P1025", "P1025-S1", "VoltTrek 100W 四口氮化镓充电器", 29900, "CNY", 2, "CONFIRMED"),
    ("P1045", "P1045-S1", "HydroFlow 316不锈钢保温水壶", 18900, "CNY", 1, "CONFIRMED"),
    ("P1039", "P1039-S1", "SolarLamp 太阳能营地灯", 12900, "CNY", 1, "CONFIRMED"),
    ("P1040", "P1040-S1", "CascadePro 铝合金折叠登山杖一对", 19900, "CNY", 1, "CANCELLED"),
    ("P1032", "P1032-S1", "Nordic 中性色陶瓷咖啡杯", 8900, "CNY", 2, "CONFIRMED"),
    ("P1057", "P1057-S1", "MiniUmbrella 五折超轻晴雨伞", 8900, "CNY", 1, "CONFIRMED"),
    ("P1028", "P1028-S1", "GlobeAdapt 全球通用转换插头", 9900, "CNY", 1, "CONFIRMED"),
    ("P1052", "P1052-S1", "TravelScale 便携行李称", 4500, "CNY", 1, "CONFIRMED"),
    ("P1048", "P1048-S1", "CampCook 钛合金炊具套装", 42900, "CNY", 1, "CONFIRMED"),
]


def ensure_tables(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS buyer_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            buyer_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            statement TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (buyer_id, kind, statement)
        );
        CREATE TABLE IF NOT EXISTS trade_orders (
            order_id TEXT PRIMARY KEY,
            buyer_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            operation_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """,
    )


def seed(db_path: Path, buyer_ids: list[str]) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    inserted_orders = 0
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA busy_timeout=10000")
        ensure_tables(db)
        for buyer_id in buyer_ids:
            for index, (product_id, sku_id, title, price, currency, quantity, status) in enumerate(
                ORDER_TEMPLATES, start=1,
            ):
                order_id = f"GBX-DEMO-{buyer_ids.index(buyer_id) + 1:02d}{index:02d}"
                operation_id = f"demo-operation-{buyer_ids.index(buyer_id) + 1:02d}-{index:02d}"
                created_at = (now - timedelta(days=index)).isoformat()
                snapshot = {
                    "order_id": order_id,
                    "buyer_id": buyer_id,
                    "status": status,
                    "total_amount_major": price / 100 * quantity,
                    "currency": currency,
                    "lines": [{
                        "product_id": product_id,
                        "sku_id": sku_id,
                        "title": title,
                        "spec": "演示款",
                        "quantity": quantity,
                        "unit_price_minor": price,
                    }],
                    "shipping_address": {
                        "country": "US",
                        "name": "Demo Buyer",
                        "line1": "100 Demo Street",
                        "city": "Seattle",
                        "postal_code": "98101",
                    },
                    "created_at": created_at,
                }
                result = db.execute(
                    "INSERT OR IGNORE INTO trade_orders"
                    " (order_id, buyer_id, session_id, operation_id, status, snapshot_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (order_id, buyer_id, f"demo-session-{buyer_id}", operation_id,
                     status, json.dumps(snapshot, ensure_ascii=False), created_at),
                )
                inserted_orders += result.rowcount
        db.commit()
    return inserted_orders


def main() -> None:
    parser = argparse.ArgumentParser(description="写入可重复的本地演示偏好和订单")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--buyer-id", default="buyer-4qw695", help="当前浏览器买家 ID")
    args = parser.parse_args()
    buyers = [args.buyer_id] if args.buyer_id == "buyer-demo" else [args.buyer_id, "buyer-demo"]
    orders = seed(args.db, buyers)
    print(f"demo buyers: {', '.join(buyers)}")
    print(f"inserted orders: {orders}")
    print(f"database: {args.db}")


if __name__ == "__main__":
    main()
