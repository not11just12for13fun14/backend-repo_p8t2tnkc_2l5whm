import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document

# -----------------------------
# FastAPI app and CORS
# -----------------------------
app = FastAPI(title="Real Estate Sales AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utils
# -----------------------------
from bson import ObjectId

def to_str_id(doc: Dict[str, Any]):
    if not doc:
        return doc
    d = dict(doc)
    if d.get("_id") is not None:
        d["id"] = str(d.pop("_id"))
    # convert datetime to iso
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d

# -----------------------------
# Personas (seed + getter)
# -----------------------------
DEFAULT_PERSONAS = [
    {
        "key": "indeciso",
        "name": "Cliente Indeciso",
        "description": "Precisa de segurança e provas sociais antes de avançar",
        "traits": ["precisa de confiança", "valoriza empatia", "evita decisão rápida"],
        "difficulty": "medium",
    },
    {
        "key": "obj_preco",
        "name": "Cliente com Objeção de Preço",
        "description": "Sente que está caro e compara muito",
        "traits": ["foca em preço", "pede desconto", "compara alternativas"],
        "difficulty": "hard",
    },
    {
        "key": "pronto",
        "name": "Cliente Pronto para Fechar",
        "description": "Já decidiu, só precisa de direcionamento para o próximo passo",
        "traits": ["objetivo", "quer agilidade", "pouca tolerância a enrolação"],
        "difficulty": "easy",
    },
]

# seed personas if collection empty
try:
    if db is not None:
        if db["persona"].count_documents({}) == 0:
            for p in DEFAULT_PERSONAS:
                create_document("persona", p)
except Exception:
    pass

# -----------------------------
# Schemas
# -----------------------------
class StartSessionRequest(BaseModel):
    seller_email: str
    persona_key: str

class ChatMessageRequest(BaseModel):
    text: str = Field(..., min_length=1)

class FinishSessionRequest(BaseModel):
    pass

# -----------------------------
# Scoring logic (simple heuristic)
# -----------------------------
RAPPORT_KEYWORDS = ["bom dia", "boa tarde", "boa noite", "tudo bem", "como vai", "prazer"]
VALUE_KEYWORDS = [
    "valor", "benefício", "localização", "valorização", "infraestrutura", "qualidade",
    "condições", "parcelamento", "financiamento", "investimento",
]
CLOSING_KEYWORDS = [
    "vamos avançar", "fechamos", "podemos agendar", "posso enviar a proposta",
    "podemos iniciar", "assinar", "fechar negócio", "aprovar cadastro",
]
OBJECTION_KEYWORDS = ["caro", "preço", "muito alto", "desconto", "tá caro", "carinho"]


def eval_message_score(text: str) -> Dict[str, float]:
    t = text.lower()
    rapport = any(k in t for k in RAPPORT_KEYWORDS)
    handles_objection = any(k in t for k in VALUE_KEYWORDS) and any(k in t for k in OBJECTION_KEYWORDS)
    closing = any(k in t for k in CLOSING_KEYWORDS)

    score = 50.0
    if rapport:
        score += 15
    if handles_objection:
        score += 20
    if closing:
        score += 15
    # cap 0..100
    score = max(0.0, min(100.0, score))

    return {
        "rapport": 100.0 if rapport else 0.0,
        "objection": 100.0 if handles_objection else 0.0,
        "closing": 100.0 if closing else 0.0,
        "overall": score,
    }


def ai_reply(persona_key: str, last_seller_text: str, turn: int) -> str:
    t = last_seller_text.lower()
    if persona_key == "indeciso":
        if turn == 1:
            return "Eu gosto, mas ainda estou inseguro. Você tem casos de clientes satisfeitos?"
        if "prova" in t or "depoimento" in t or "cases" in t:
            return "Legal. Você poderia me mostrar alguns depoimentos e me explicar por que essa é a melhor opção para mim?"
        return "Entendo, mas talvez eu deva esperar um pouco. O que você acha?"
    if persona_key == "obj_preco":
        if "desconto" in t or "condi" in t or "financ" in t:
            return "Se tiver boas condições e eu enxergar valor, podemos seguir. Qual seria a proposta?"
        if turn == 1:
            return "Achei o preço um pouco alto comparado a outras opções."
        return "Está caro. Qual é o diferencial real?"
    if persona_key == "pronto":
        if "agendar" in t or "proposta" in t or "fechar" in t or "vamos" in t:
            return "Perfeito, pode me enviar agora."
        return "Já gostei. Qual o próximo passo para avançarmos?"
    # default
    return "Poderia explicar melhor?"

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Real Estate Sales AI Backend"}


@app.get("/api/personas")
def get_personas():
    if db is None:
        return DEFAULT_PERSONAS
    docs = [to_str_id(d) for d in db["persona"].find({}).limit(50)]
    # ensure at least defaults
    if not docs:
        return DEFAULT_PERSONAS
    # don't expose internal id for personas in UI needs
    for d in docs:
        d.pop("id", None)
    return docs


@app.post("/api/sessions/start")
def start_session(payload: StartSessionRequest):
    if db is None:
        raise HTTPException(500, "Database not available")

    persona = db["persona"].find_one({"key": payload.persona_key})
    if not persona:
        # allow defaults without db seed
        keys = [p["key"] for p in DEFAULT_PERSONAS]
        if payload.persona_key not in keys:
            raise HTTPException(400, "Persona not found")

    session_doc = {
        "seller_email": payload.seller_email,
        "persona_key": payload.persona_key,
        "status": "active",
        "current_score": 0.0,
        "total_messages": 0,
        "messages": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    _id = db["roleplaysession"].insert_one(session_doc).inserted_id
    doc = db["roleplaysession"].find_one({"_id": _id})
    return to_str_id(doc)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if db is None:
        raise HTTPException(500, "Database not available")
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")
    doc = db["roleplaysession"].find_one({"_id": oid})
    if not doc:
        raise HTTPException(404, "Session not found")
    return to_str_id(doc)


@app.post("/api/sessions/{session_id}/message")
def send_message(session_id: str, payload: ChatMessageRequest):
    if db is None:
        raise HTTPException(500, "Database not available")
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")

    session = db["roleplaysession"].find_one({"_id": oid})
    if not session:
        raise HTTPException(404, "Session not found")
    if session.get("status") != "active":
        raise HTTPException(400, "Session already finished")

    # append seller message
    messages: List[Dict[str, Any]] = session.get("messages", [])
    messages.append({"role": "seller", "text": payload.text, "ts": datetime.now(timezone.utc).isoformat()})

    # scoring update
    metrics = eval_message_score(payload.text)
    # rolling score: weighted by number of seller turns
    seller_turns = len([m for m in messages if m["role"] == "seller"])
    new_overall = (session.get("current_score", 0.0) * (seller_turns - 1) + metrics["overall"]) / max(1, seller_turns)

    # AI reply
    reply_text = ai_reply(session.get("persona_key"), payload.text, seller_turns)
    messages.append({"role": "ai", "text": reply_text, "ts": datetime.now(timezone.utc).isoformat()})

    db["roleplaysession"].update_one(
        {"_id": oid},
        {
            "$set": {
                "messages": messages,
                "current_score": new_overall,
                "total_messages": len(messages),
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )

    updated = db["roleplaysession"].find_one({"_id": oid})
    out = to_str_id(updated)
    out["last_metrics"] = metrics
    return out


@app.post("/api/sessions/{session_id}/finish")
def finish_session(session_id: str, _: FinishSessionRequest):
    if db is None:
        raise HTTPException(500, "Database not available")
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(400, "Invalid session id")

    session = db["roleplaysession"].find_one({"_id": oid})
    if not session:
        raise HTTPException(404, "Session not found")

    db["roleplaysession"].update_one({"_id": oid}, {"$set": {"status": "finished", "updated_at": datetime.now(timezone.utc)}})

    # store score snapshot
    score_doc = {
        "session_id": str(oid),
        "seller_email": session["seller_email"],
        "persona_key": session["persona_key"],
        "final_score": session.get("current_score", 0.0),
        "created_at": datetime.now(timezone.utc),
    }
    db["sessionscore"].insert_one(score_doc)

    updated = db["roleplaysession"].find_one({"_id": oid})
    return to_str_id(updated)


@app.get("/api/history")
def history(seller_email: str):
    if db is None:
        raise HTTPException(500, "Database not available")
    cur = db["sessionscore"].find({"seller_email": seller_email}).sort("created_at", -1).limit(50)
    return [to_str_id(d) for d in cur]


@app.get("/api/leaderboard")
def leaderboard(period: Literal["7d", "30d", "all"] = "30d"):
    if db is None:
        raise HTTPException(500, "Database not available")

    since = None
    if period == "7d":
        since = datetime.now(timezone.utc) - timedelta(days=7)
    elif period == "30d":
        since = datetime.now(timezone.utc) - timedelta(days=30)

    match_stage = {"$match": {}}
    if since:
        match_stage["$match"]["created_at"] = {"$gte": since}

    pipeline = [
        match_stage,
        {
            "$group": {
                "_id": "$seller_email",
                "avg_score": {"$avg": "$final_score"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"avg_score": -1}},
        {"$limit": 20},
    ]

    results = list(db["sessionscore"].aggregate(pipeline))
    out = [
        {"seller_email": r["_id"], "avg_score": round(r.get("avg_score", 0.0), 1), "sessions": r.get("count", 0)}
        for r in results
    ]
    return out


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
