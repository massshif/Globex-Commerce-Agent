"""Durable approval boundary for local orders and stock.

All inventory changes, order rows, and operation outcomes share one SQLite
transaction. A repeated approval returns the recorded outcome.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.domain.catalog.product import Product
from app.domain.order.address import Address
from app.infrastructure.persistence.seed_products import build_seed_products


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeConflict(ValueError):
    pass


class TradeStore:
    def __init__(self, database: Path, products: list[Product] | None = None) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        # The composition root injects the catalog snapshot used by search and
        # ordering. The seed fallback exists only for isolated unit tests.
        self._products = {p.product_id: p for p in (products or build_seed_products())}
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS trade_stock (
                    sku_id TEXT PRIMARY KEY, product_id TEXT NOT NULL,
                    stock INTEGER NOT NULL CHECK(stock >= 0));
                CREATE TABLE IF NOT EXISTS trade_confirmations (
                    operation_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, action TEXT NOT NULL,
                    draft_json TEXT NOT NULL, draft_hash TEXT NOT NULL,
                    status TEXT NOT NULL, outcome_json TEXT,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trade_orders (
                    order_id TEXT PRIMARY KEY, buyer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trade_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL, buyer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, type TEXT NOT NULL,
                    payload_json TEXT NOT NULL, occurred_at TEXT NOT NULL);
            """)
            for product in self._products.values():
                for sku in product.skus:
                    db.execute("INSERT OR IGNORE INTO trade_stock VALUES (?,?,?)",
                               (sku.sku_id, product.product_id, sku.stock))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def prepare_order(self, buyer_id: str, session_id: str, items: list[dict], address: dict) -> dict:
        if not buyer_id or not session_id or not items:
            raise ValueError("buyer, session and items are required")
        Address(**address)  # validate before opening the transaction
        products = self._products
        lines: list[dict] = []
        currency: str | None = None
        total_minor = 0
        with self._connect() as db:
            for item in items:
                product = products.get(item["product_id"])
                sku = product.find_sku(item["sku_id"]) if product else None
                quantity = int(item.get("quantity", 1))
                if sku is None or quantity <= 0:
                    raise ValueError("商品、SKU 或数量无效")
                row = db.execute("SELECT stock FROM trade_stock WHERE sku_id=? AND product_id=?",
                                 (sku.sku_id, product.product_id)).fetchone()
                if row is None or row["stock"] < quantity:
                    raise ValueError(f"库存不足：{sku.sku_id}")
                if currency and sku.price.currency != currency:
                    raise ValueError("订单行币种不一致")
                currency = sku.price.currency
                total_minor += sku.price.amount_in_minor_units * quantity
                lines.append({"product_id": product.product_id, "sku_id": sku.sku_id,
                              "title": product.title, "spec": sku.spec, "quantity": quantity,
                              "unit_price_minor": sku.price.amount_in_minor_units})
            draft = {"items": lines, "shipping_address": address,
                     "total_amount_minor": total_minor, "currency": currency}
            digest = hashlib.sha256(_json(draft).encode()).hexdigest()
            operation_id = f"op-{uuid.uuid4().hex}"
            db.execute("INSERT INTO trade_confirmations VALUES (?,?,?,?,?,?,?,?,?)",
                       (operation_id, buyer_id, session_id, "place_order", _json(draft), digest,
                        "pending", None, _now()))
            self._event(db, operation_id, buyer_id, session_id, "confirmation.prepared", draft)
        return {"operation_id": operation_id, "draft_hash": digest, "action": "place_order",
                "buyer_id": buyer_id, "session_id": session_id, "draft": draft, "status": "pending"}

    def prepare_cancel(self, buyer_id: str, session_id: str, order_id: str, reason: str) -> dict:
        with self._connect() as db:
            order = db.execute("SELECT * FROM trade_orders WHERE order_id=? AND buyer_id=?", (order_id, buyer_id)).fetchone()
            if order is None or order["status"] != "CONFIRMED":
                raise ValueError("订单不存在或不可取消")
            draft = {"order_id": order_id, "reason": reason, "order_snapshot": json.loads(order["snapshot_json"])}
            digest = hashlib.sha256(_json(draft).encode()).hexdigest()
            operation_id = f"op-{uuid.uuid4().hex}"
            db.execute("INSERT INTO trade_confirmations VALUES (?,?,?,?,?,?,?,?,?)",
                       (operation_id, buyer_id, session_id, "cancel_order", _json(draft), digest,
                        "pending", None, _now()))
            self._event(db, operation_id, buyer_id, session_id, "confirmation.prepared", draft)
        return {"operation_id": operation_id, "draft_hash": digest, "action": "cancel_order",
                "buyer_id": buyer_id, "session_id": session_id, "draft": draft, "status": "pending"}

    def get_confirmation(self, operation_id: str, buyer_id: str, session_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM trade_confirmations WHERE operation_id=? AND buyer_id=? AND session_id=?",
                             (operation_id, buyer_id, session_id)).fetchone()
        if row is None:
            return None
        return {"operation_id": operation_id, "buyer_id": buyer_id, "session_id": session_id,
                "action": row["action"], "draft": json.loads(row["draft_json"]),
                "draft_hash": row["draft_hash"], "status": row["status"],
                "outcome": json.loads(row["outcome_json"]) if row["outcome_json"] else None}

    def decide(self, operation_id: str, buyer_id: str, session_id: str,
               draft_hash: str, approved: bool) -> dict:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM trade_confirmations WHERE operation_id=? AND buyer_id=? AND session_id=?",
                             (operation_id, buyer_id, session_id)).fetchone()
            if row is None or row["draft_hash"] != draft_hash:
                raise TradeConflict("确认不存在、买家/会话不匹配或金额快照已变")
            if row["status"] != "pending":
                return json.loads(row["outcome_json"])
            draft = json.loads(row["draft_json"])
            if not approved:
                outcome = {"operation_id": operation_id, "status": "rejected"}
                status = "rejected"
            elif row["action"] == "place_order":
                # Revalidate the authoritative product/SKU data and stock at commit time.
                products = self._products
                for line in draft["items"]:
                    product = products.get(line["product_id"])
                    sku = product.find_sku(line["sku_id"]) if product else None
                    if sku is None or sku.price.amount_in_minor_units != line["unit_price_minor"]:
                        raise TradeConflict("商品规格或价格已经变化，请重新确认")
                    changed = db.execute("UPDATE trade_stock SET stock=stock-? WHERE sku_id=? AND product_id=? AND stock>=?",
                                         (line["quantity"], line["sku_id"], line["product_id"], line["quantity"]))
                    if changed.rowcount != 1:
                        raise TradeConflict(f"库存已变化，请重新确认：{line['sku_id']}")
                order_id = f"GBX-{uuid.uuid4().hex[:12].upper()}"
                snapshot = {"order_id": order_id, "buyer_id": buyer_id, "status": "CONFIRMED",
                            "total_amount_major": draft["total_amount_minor"] / 100,
                            "currency": draft["currency"], "lines": draft["items"],
                            "shipping_address": draft["shipping_address"], "created_at": _now()}
                db.execute("INSERT INTO trade_orders VALUES (?,?,?,?,?,?,?)",
                           (order_id, buyer_id, session_id, operation_id, "CONFIRMED", _json(snapshot), _now()))
                outcome = {"operation_id": operation_id, "status": "confirmed", "order": snapshot}
                status = "confirmed"
            else:
                order = db.execute("SELECT * FROM trade_orders WHERE order_id=? AND buyer_id=?",
                                   (draft["order_id"], buyer_id)).fetchone()
                if order is None or order["status"] != "CONFIRMED":
                    raise TradeConflict("订单状态已变化，请重新确认")
                snapshot = json.loads(order["snapshot_json"])
                for line in snapshot["lines"]:
                    db.execute("UPDATE trade_stock SET stock=stock+? WHERE sku_id=?",
                               (line["quantity"], line["sku_id"]))
                snapshot.update(status="CANCELLED", cancel_reason=draft["reason"])
                db.execute("UPDATE trade_orders SET status='CANCELLED', snapshot_json=? WHERE order_id=?",
                           (_json(snapshot), draft["order_id"]))
                outcome = {"operation_id": operation_id, "status": "cancelled", "order": snapshot}
                status = "cancelled"
            db.execute("UPDATE trade_confirmations SET status=?, outcome_json=? WHERE operation_id=?",
                       (status, _json(outcome), operation_id))
            self._event(db, operation_id, buyer_id, session_id, f"confirmation.{status}", outcome)
        return outcome

    def list_orders(self, buyer_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT snapshot_json FROM trade_orders WHERE buyer_id=? ORDER BY created_at DESC", (buyer_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_order(self, order_id: str, buyer_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT snapshot_json FROM trade_orders WHERE order_id=? AND buyer_id=?", (order_id, buyer_id)).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, buyer_id: str, session_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM trade_events WHERE buyer_id=? AND session_id=? ORDER BY sequence",
                              (buyer_id, session_id)).fetchall()
        return [{"sequence": row["sequence"], "type": row["type"],
                 "payload": json.loads(row["payload_json"]), "occurred_at": row["occurred_at"]}
                for row in rows]

    @staticmethod
    def _event(db: sqlite3.Connection, operation_id: str, buyer_id: str, session_id: str,
               event_type: str, payload: dict) -> None:
        db.execute("INSERT INTO trade_events(operation_id,buyer_id,session_id,type,payload_json,occurred_at) VALUES (?,?,?,?,?,?)",
                   (operation_id, buyer_id, session_id, event_type, _json(payload), _now()))
