"""Small in-process BM25 index for the authoritative product catalog."""
from __future__ import annotations

import math
import re
from collections import Counter

from app.domain.catalog.product import Product

_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)


def terms(text: str) -> list[str]:
    result: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group()
        if "\u4e00" <= token[0] <= "\u9fff" and len(token) > 1:
            result.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            result.append(token)
    return result


class BM25ProductIndex:
    def __init__(self, products: list[Product], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._products = products
        self._documents = [Counter(terms(p.searchable_text() + " " + " ".join(s.spec for s in p.skus))) for p in products]
        self._lengths = [sum(doc.values()) for doc in self._documents]
        self._average = sum(self._lengths) / len(products) if products else 1.0
        self._df = Counter(token for doc in self._documents for token in doc)
        self._k1, self._b = k1, b

    def search(self, query: str, top_n: int) -> list[tuple[float, Product]]:
        query_terms = set(terms(query))
        scored: list[tuple[float, Product]] = []
        count = len(self._products)
        for product, document, length in zip(self._products, self._documents, self._lengths):
            score = 0.0
            for token in query_terms:
                tf = document[token]
                if not tf:
                    continue
                idf = math.log(1 + (count - self._df[token] + 0.5) / (self._df[token] + 0.5))
                score += idf * tf * (self._k1 + 1) / (tf + self._k1 * (1 - self._b + self._b * length / self._average))
            # Product attributes in the query are stronger evidence than a
            # generic title/category hit (for example “抗造” vs “三件套”).
            searchable = product.searchable_text().lower()
            for attribute in ("抗造", "耐磨", "轻便", "无塑料", "防水", "降噪"):
                if attribute in query.lower() and attribute in searchable:
                    score += 4.0
            if score > 0:
                scored.append((score, product))
        scored.sort(key=lambda item: (-item[0], item[1].product_id))
        return scored[:top_n]
