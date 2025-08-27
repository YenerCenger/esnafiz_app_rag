# api/endpoints/chat.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from services.dependencies import get_service_manager
from services.retrieval.hybrid import hybrid_search
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, status
import jwt
from datetime import datetime
from core.config import get_settings
import json
from textwrap import dedent
import re

router = APIRouter(prefix="/chat", tags=["chat"])
security = HTTPBearer(auto_error=False)
settings = get_settings()

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    k: int = Field(8, ge=1, le=30)
    intent: str | None = Field(default=None, description="inventory | sales | general (auto)")

def _decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        exp = payload.get("exp")
        if isinstance(exp, (int, float)) and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _get_store_context(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    payload = _decode_token(credentials.credentials)
    store_id = payload.get("store_id")
    user_id = payload.get("user_id")
    if not store_id or not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return {"store_id": store_id, "user_id": user_id}


@router.post("", summary="RAG chat (basit)")
async def chat_endpoint(payload: ChatRequest, sm=Depends(get_service_manager), ctx: Dict[str, Any] = Depends(_get_store_context)):
    # Rate limit
    rate_key = f"{ctx.get('store_id')}:{ctx.get('user_id')}"
    if not await sm.check_rate_limit(rate_key, settings.rate_per_minute):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Hız limiti aşıldı")

    # Cache kontrolü (deterministik anahtar)
    cache_key = sm.make_cache_key(
        "chat",
        [str(ctx.get('store_id') or ''), payload.query, str(payload.k)],
    )
    cached = await sm.cache_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    # Intent yönlendirme
    q_lower = payload.query.lower()
    def detect_barcode(text: str) -> str | None:
        m = re.findall(r"\b\d{8,14}\b", text or "")
        return m[0] if m else None

    # Genişletilmiş niyet tespiti (barkod sadece sinyal, zorunlu değil)
    if payload.intent:
        intent = payload.intent
    else:
        if any(k in q_lower for k in ["fiyat ne olmalı", "fiyatı ne olmalı", "kaçtan sat", "satış fiyatı ne", "ne kadar olsun", "price suggestion"]):
            intent = "price_suggestion"
        elif any(k in q_lower for k in ["karşılaştır", "fiyat karşılaştır"]):
            intent = "compare_prices"
        elif any(k in q_lower for k in ["indirim yap", "%", "yüzde indir", "kampanya"]):
            intent = "promo_guard"
        elif any(k in q_lower for k in ["paket yap", "bundle", "set fiyat"]):
            intent = "bundle_price"
        elif any(k in q_lower for k in ["benzer ürün", "alternatif", "şuna benzer"]):
            intent = "new_item_similar"
        elif any(k in q_lower for k in ["reorder point", "yeniden sipariş noktası", "rop"]):
            intent = "reorder_point"
        elif any(k in q_lower for k in ["ölü stok", "raf", "satılmayan"]):
            intent = "dead_stock"
        elif any(k in q_lower for k in ["düşük stok", "bitmek üzere"]):
            intent = "low_stock_alerts"
        elif any(k in q_lower for k in ["son 7 gün", "son 30 gün", "ciro", "satış özeti", "satış"]):
            intent = "sales_summary"
        elif any(k in q_lower for k in ["abc analizi", "abc analysis"]):
            intent = "abc_analysis"
        elif any(k in q_lower for k in ["iade", "kargo", "sss", "sık sorulan", "policy", "politika"]):
            intent = "faq_rag"
        elif any(k in q_lower for k in ["stok", "envanter", "kaç adet", "ürün"]):
            intent = "inventory"
        elif any(k in q_lower for k in ["toptancı", "wholesale", "tedarikçi", "en ucuz", "nerede ucuz", "en uygun"]):
            intent = "wholesale"
        else:
            intent = "general"

    # Firestore intentleri
    if intent == "inventory":
        items = await sm.get_store_items(ctx.get("store_id"))
        if not items:
            return {"answer": "Bu mağaza için envanter verisi bulunamadı.", "sources": [], "metrics": {"intent": intent}, "store": ctx}
        # basit özet
        total = len(items)
        zero = sum(1 for it in items if (it.get("stock") or 0) == 0)
        low = sum(1 for it in items if 0 < (it.get("stock") or 0) <= 5)
        # kullanıcı sorgusuna göre eşleşen ilk 5 ürün
        matched = [it for it in items if (payload.query.strip().lower() in str(it.get("name","")) .lower())]
        preview = "\n".join([f"- {it.get('name')} ({it.get('stock')} adet)" for it in matched[:5]]) if matched else ""
        answer = f"Toplam ürün: {total}. Tükenen: {zero}. Düşük stok: {low}." + (f"\n\nEşleşenler:\n{preview}" if preview else "")
        return {"answer": answer, "sources": [{"type": "inventory", "count": total}], "metrics": {"intent": intent}, "store": ctx}

    if intent == "sales_summary":
        sales = await sm.get_store_sales(ctx.get("store_id"), days=30)
        if not sales:
            return {"answer": "Son 30 günde satış verisi bulunamadı.", "sources": [], "metrics": {"intent": intent}, "store": ctx}
        total_qty = sum(s.get("quantity", 0) for s in sales)
        total_rev = sum((s.get("quantity", 0) * s.get("price", 0.0)) for s in sales)
        answer = f"Toplam satış: {total_qty} adet. Toplam ciro: {total_rev:.2f}₺."
        return {"answer": answer, "sources": [{"type": "sales", "count": len(sales)}], "metrics": {"intent": intent}, "store": ctx}

    if intent == "abc_analysis":
        sales = await sm.get_store_sales(ctx.get("store_id"), days=30)
        if not sales:
            return {"answer": "ABC analizi için yeterli satış verisi yok.", "sources": [], "metrics": {"intent": intent}, "store": ctx}
        rev_by_prod: dict[str, float] = {}
        for s in sales:
            name = str(s.get("product_name", "Unknown"))
            rev_by_prod[name] = rev_by_prod.get(name, 0.0) + float(s.get("price", 0.0)) * int(s.get("quantity", 1))
        total = sum(rev_by_prod.values()) or 1.0
        sorted_items = sorted(rev_by_prod.items(), key=lambda x: x[1], reverse=True)
        cum = 0.0
        A = B = C = 0
        for _, amt in sorted_items:
            cum += amt / total
            if cum <= 0.8:
                A += 1
            elif cum <= 0.95:
                B += 1
            else:
                C += 1
        answer = f"ABC: A={A} ürün, B={B}, C={C}."
        return {"answer": answer, "sources": [{"type": "sales", "analysis": "abc"}], "metrics": {"intent": intent}, "store": ctx}

    if intent == "low_stock_alerts":
        items = await sm.get_store_items(ctx.get("store_id"))
        low = [it for it in items if (it.get("stock") or 0) <= 3]
        if not low:
            return {"answer": "Eşik altı stok yok.", "sources": [{"type": "inventory"}], "metrics": {"intent": intent}, "store": ctx}
        lines = "\n".join([f"- {it.get('name')} : {it.get('stock')}" for it in low[:20]])
        return {"answer": "Bitmek üzere olanlar:\n" + lines, "sources": [{"type": "inventory"}], "metrics": {"intent": intent}, "store": ctx}

    if intent == "wholesale":
        # Basit: barkod sinyal ise barkodla, değilse RAG ile toptan bağlam araması
        bar = detect_barcode(payload.query)
        if sm.firestore and bar:
            loop = __import__('asyncio').get_event_loop()
            def _fetch():
                return list(sm.firestore.collection("wholesale").where("barcode", "==", str(bar)).limit(20).stream())
            docs = await loop.run_in_executor(sm.thread_pool, _fetch)
            if docs:
                lines = []
                for d in docs[:8]:
                    x = d.to_dict()
                    lines.append(f"- {x.get('name')} ({x.get('supplier')}): {x.get('wholesalePrice')}₺ MOQ {x.get('minOrderQty')}")
                return {"answer": "\n".join(lines), "sources": [{"type": "wholesale", "by": "barcode"}], "metrics": {"intent": intent}, "store": ctx}
        # Aksi halde genel moda düş

    # Vektör araması ile bağlamları getir (general intent)
    results = await hybrid_search(sm, payload.query, k=payload.k, store_id=ctx.get("store_id"))
    context_text = "\n---\n".join([(r.get("text") or "") for r in results])

    # LLM prompt
    prompt = dedent(f"""
    Sen Esnafız AI asistanısın. Türkiye'deki esnaflara yardımcı ol.

    Mağaza: {ctx.get('store_id')}
    Kullanıcı Sorusu: {payload.query}

    Bağlam:
    {context_text}

    Kurallar:
    - Yalnızca bağlamdaki bilgilere dayan
    - Türkçe, kısa ve net cevap ver
    - Bağlamda yoksa "Bu konuda elimde yeterli veri yok" de
    """
    ).strip()

    llm_answer = await sm.call_llm(prompt)
    answer = llm_answer or "Bu konuda elimde yeterli veri yok."
    sources = [
        {
            "id": r.get("id"),
            "score": round(float(r.get("score", 0.0)), 3),
            "doc_id": (r.get("payload") or {}).get("doc_id"),
            "collection": (r.get("payload") or {}).get("collection"),
        }
        for r in results
    ]
    response = {
        "answer": answer,
        "sources": sources,
        "metrics": {"k": payload.k, "retrieval_count": len(results)},
        "store": ctx,
    }

    # Cache yaz
    try:
        await sm.cache_setex(cache_key, settings.cache_ttl, json.dumps(response, ensure_ascii=False))
    except Exception:
        pass

    return response
