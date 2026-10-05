import { useEffect, useRef, useState } from "react";
import { HttpAgent, randomUUID } from "@ag-ui/client";
import EventTimeline from "./components/EventTimeline";
import ProductCards from "./components/ProductCards";
import type { TradeEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

function loadOrCreate(key: string, prefix: string): string {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const created = `${prefix}-${Math.random().toString(36).slice(2, 8)}`;
  localStorage.setItem(key, created);
  return created;
}

interface Turn {
  role: "buyer" | "agent";
  text: string;
}

export default function App() {
  const [sessionId] = useState(() => loadOrCreate("globex.session", "web"));
  const [buyerId] = useState(() => loadOrCreate("globex.buyer", "buyer"));
  const [events, setEvents] = useState<TradeEvent[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const [page, setPage] = useState<"chat" | "products" | "preferences" | "orders">("chat");
  const [preferences, setPreferences] = useState<any[]>([]);
  const [orders, setOrders] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const agAgent = useRef(new HttpAgent({ url: `${API_BASE}/ag-ui/agent`, headers: { "X-Buyer-Id": "browser" } }));

  // WS 订阅：按会话接收 Agent 过程事件（StrictMode 下会双次挂载，用 closed 标记避免早关告警）
  useEffect(() => {
    let closed = false;
    let retryTimer: number | undefined;

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(`${WS_BASE}/commerce/events`);
      wsRef.current = ws;
      ws.onopen = () => {
        if (closed) {
          ws.close();
          return;
        }
        ws.send(JSON.stringify({ shopping_session_id: sessionId }));
        setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          // 断线重连，避免长任务期间丢事件
          retryTimer = window.setTimeout(connect, 1500);
        }
      };
      ws.onmessage = (message) => {
        const event: TradeEvent = JSON.parse(message.data);
        if (event.type === "token.delta") {
          setStreaming((prev) => prev + (event.payload.token ?? ""));
          return;
        }
        setEvents((prev) => [...prev, event]);
        if (event.type === "final.result") {
          setStreaming("");
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, [sessionId]);

  const submit = async () => {
    const query = input.trim();
    if (!query || busy) return;
    setInput("");
    setBusy(true);
    setTurns((prev) => [...prev, { role: "buyer", text: query }]);
    try {
      const agent = agAgent.current;
      agent.messages.push({ id: randomUUID(), role: "user", content: query });
      let answer = "";
      await agent.runAgent({}, {
        onTextMessageContentEvent: ({ event }) => { answer += event.delta ?? ""; setStreaming(answer); },
        onTextMessageEndEvent: () => { setTurns((prev) => [...prev, { role: "agent", text: answer }]); setStreaming(""); },
      });
    } catch (error) {
      setTurns((prev) => [...prev, { role: "agent", text: `[error] 请求失败：${error}` }]);
    } finally {
      setBusy(false);
    }
  };

  const loadPage = async (target: typeof page) => {
    setPage(target);
    if (target === "preferences") {
      const response = await fetch(`${API_BASE}/commerce/preferences/${buyerId}`);
      if (response.ok) setPreferences(await response.json());
    }
    if (target === "orders") {
      const response = await fetch(`${API_BASE}/commerce/orders?buyer_id=${encodeURIComponent(buyerId)}`);
      if (response.ok) setOrders(await response.json());
    }
  };

  return (
    <div className="layout">
      <header>
        <h1>Globex 跨境购物助手</h1>
        <div className="meta">
          <span>会话 {sessionId}</span>
          <span>买家 {buyerId}</span>
          <span className={connected ? "dot on" : "dot off"}>{connected ? "事件流已连接" : "事件流断开"}</span>
        </div>
        <nav className="nav-tabs" aria-label="买家功能">
          {(["chat", "products", "preferences", "orders"] as const).map((item) => (
            <button key={item} className={page === item ? "active" : ""} onClick={() => void loadPage(item)}>
              {{ chat: "对话", products: "商品卡", preferences: "Skill 与偏好", orders: "我的订单" }[item]}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {page === "chat" && <section className="chat">
          <div className="turns">
            {turns.map((turn, index) => (
              <div key={index} className={`turn ${turn.role}`}>
                <div className="who">{turn.role === "buyer" ? "我" : "Globex"}</div>
                <div className="text">{turn.text}</div>
              </div>
            ))}
            {streaming && (
              <div className="turn agent streaming">
                <div className="who">Globex</div>
                <div className="text">{streaming}</div>
              </div>
            )}
            {busy && !streaming && <div className="hint">Agent 正在处理……</div>}
          </div>

          <ProductCards events={events} />

          <div className="composer">
            <textarea
              value={input}
              placeholder="例如：我人在美国，250 美元预算买个降噪耳机寄美国，到手价多少？"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            <button onClick={() => void submit()} disabled={busy || !input.trim()}>
              {busy ? "处理中" : "发送"}
            </button>
          </div>
        </section>}
        {page === "products" && <section className="page-panel"><h2>商品卡</h2><ProductCards events={events} /><p className="hint">商品卡来自最近一次检索结果。</p></section>}
        {page === "preferences" && <section className="page-panel"><h2>Skill 与偏好</h2><p className="hint">Agent 会在后续会话中自动注入这些偏好。</p><ul className="data-list">{preferences.map((item) => <li key={`${item.kind}-${item.statement}`}><b>{item.kind === "like" ? "喜欢" : "避开"}</b>：{item.statement}</li>)}{preferences.length === 0 && <li>暂无已保存偏好</li>}</ul></section>}
        {page === "orders" && <section className="page-panel"><h2>我的订单</h2><ul className="data-list">{orders.map((order) => <li key={order.order_id}><b>{order.order_id}</b> · {order.status} · {order.total_amount_major} {order.currency}</li>)}{orders.length === 0 && <li>暂无订单</li>}</ul></section>}

        <EventTimeline events={events} />
      </main>
    </div>
  );
}
