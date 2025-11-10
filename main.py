import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
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
        "disc_profile": "S",
        "triggers": ["provas sociais", "garantia", "tempo para decidir"],
    },
    {
        "key": "obj_preco",
        "name": "Cliente com Objeção de Preço",
        "description": "Sente que está caro e compara muito",
        "traits": ["foca em preço", "pede desconto", "compara alternativas"],
        "difficulty": "hard",
        "disc_profile": "C",
        "triggers": ["valor agregado", "comparativos", "ROI"],
    },
    {
        "key": "pronto",
        "name": "Cliente Pronto para Fechar",
        "description": "Já decidiu, só precisa de direcionamento para o próximo passo",
        "traits": ["objetivo", "quer agilidade", "pouca tolerância a enrolação"],
        "difficulty": "easy",
        "disc_profile": "D",
        "triggers": ["próximos passos claros", "agilidade"],
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
# Auth (simple header-based stub)
# -----------------------------
class AuthUser(BaseModel):
    email: str
    role: Literal["seller", "manager", "admin"] = "seller"
    team: Optional[str] = None


def get_current_user(x_user: Optional[str] = Header(None, alias="X-User")) -> AuthUser:
    # For MVP: accept X-User header with "email|role|team" or just email
    # In real implementation, replace with JWT auth.
    if x_user:
        parts = x_user.split("|")
        if len(parts) == 3:
            return AuthUser(email=parts[0], role=parts[1], team=parts[2])
        return AuthUser(email=x_user)
    return AuthUser(email="anonymous@example.com")


# -----------------------------
# Schemas
# -----------------------------
class StartSessionRequest(BaseModel):
    seller_email: str
    persona_key: str
    weights: Optional[Dict[str, float]] = None


class ChatMessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


class FinishSessionRequest(BaseModel):
    pass


class ScoreConfigPayload(BaseModel):
    scope: Literal["global", "team", "user"] = "global"
    team: Optional[str] = None
    email: Optional[str] = None
    weights: Dict[str, float]


class RegisterPayload(BaseModel):
    name: str
    email: str
    team: Optional[str] = None
    role: Literal["seller", "manager", "admin"] = "seller"


# -----------------------------
# Scoring logic
# -----------------------------
RAPPORT_KEYWORDS = ["bom dia", "boa tarde", "boa noite", "tudo bem", "como vai", "prazer"]
VALUE_KEYWORDS = [
    "valor",
    "benefício",
    "localização",
    "valorização",
    "infraestrutura",
    "qualidade",
    "condições",
    "parcelamento",
    "financiamento",
    "investimento",
]
CLOSING_KEYWORDS = [
    "vamos avançar",
    "fechamos",
    "podemos agendar",
    "posso enviar a proposta",
    "podemos iniciar",
    "assinar",
    "fechar negócio",
    "aprovar cadastro",
]
OBJECTION_KEYWORDS = ["caro", "preço", "muito alto", "desconto", "tá caro", "carinho"]
DISCOVERY_KEYWORDS = ["por que", "o que é mais importante", "orçamento", "prazo", "decisor", "necessidade"]


def eval_message_subscores(text: str) -> Dict[str, float]:
    t = text.lower()
    rapport = any(k in t for k in RAPPORT_KEYWORDS)
    discovery = any(k in t for k in DISCOVERY_KEYWORDS)
    handles_objection = any(k in t for k in VALUE_KEYWORDS) and any(
        k in t for k in OBJECTION_KEYWORDS
    )
    closing = any(k in t for k in CLOSING_KEYWORDS)
    return {
        "rapport": 100.0 if rapport else 0.0,
        "discovery": 100.0 if discovery else 0.0,
        "objection": 100.0 if handles_objection else 0.0,
        "closing": 100.0 if closing else 0.0,
    }


def weighted_overall(sub: Dict[str, float], weights: Dict[str, float]):
    # normalize weights
    total_w = sum(weights.values()) or 1.0
    w = {k: v / total_w for k, v in weights.items()}
    return sum(sub.get(k, 0.0) * w.get(k, 0.0) for k in w.keys())


# -----------------------------
# LLM integration (OpenAI compatible)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None


def generate_ai_reply(persona: Dict[str, Any], history: List[Dict[str, Any]], seller_text: str) -> str:
    # If no API key, fallback to heuristic reply
    if openai_client is None:
        # simple deterministic fallback
        turn = len([m for m in history if m.get("role") == "seller"]) + 1
        return ai_reply(persona.get("key"), seller_text, turn)

    system_prompt = (
        "Você é um cliente em um roleplay de vendas imobiliárias. Responda de forma realista, breve (1-2 frases),"
        " seguindo o perfil DISC e gatilhos da persona, levantando objeções quando adequado."
    )
    persona_desc = f"Persona: {persona.get('name')} | Perfil DISC: {persona.get('disc_profile')} | Traços: {', '.join(persona.get('traits', []))} | Gatilhos: {', '.join(persona.get('triggers', []))}"

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{persona_desc}"},
    ]
    # include last 6 exchanges
    for m in history[-12:]:
        role = "assistant" if m.get("role") == "ai" else "user"
        msgs.append({"role": role, "content": m.get("text", "")})
    msgs.append({"role": "user", "content": seller_text})

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.7,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # fallback safe behavior
        turn = len([m for m in history if m.get("role") == "seller"]) + 1
        return ai_reply(persona.get("key"), seller_text, turn)


# Backward-compatible heuristic reply

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


@app.post("/api/register")
def register(payload: RegisterPayload):
    if db is None:
        raise HTTPException(500, "Database not available")
    doc = {
        "name": payload.name,
        "email": payload.email,
        "team": payload.team,
        "role": payload.role,
        "is_active": True,
        "updated_at": datetime.now(timezone.utc),
    }
    db["seller"].update_one({"email": payload.email}, {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}}, upsert=True)
    return {"status": "ok"}


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
        # construct persona dict from defaults
        persona = next(p for p in DEFAULT_PERSONAS if p["key"] == payload.persona_key)

    default_weights = {"rapport": 0.3, "discovery": 0.2, "objection": 0.3, "closing": 0.2}

    session_doc = {
        "seller_email": payload.seller_email,
        "persona_key": payload.persona_key,
        "status": "active",
        "current_score": 0.0,
        "total_messages": 0,
        "scoring_weights": payload.weights or get_weight_config_for_user(payload.seller_email) or default_weights,
        "premium_unlocked": False,
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
    subs = eval_message_subscores(payload.text)
    weights = session.get("scoring_weights", {"rapport": 0.3, "discovery": 0.2, "objection": 0.3, "closing": 0.2})
    overall = weighted_overall(subs, weights)

    # rolling score: weighted by number of seller turns
    seller_turns = len([m for m in messages if m["role"] == "seller"])
    new_overall = (session.get("current_score", 0.0) * (seller_turns - 1) + overall) / max(1, seller_turns)

    # AI reply via LLM or fallback
    persona = db["persona"].find_one({"key": session.get("persona_key")}) or next(
        (p for p in DEFAULT_PERSONAS if p["key"] == session.get("persona_key")), {}
    )
    reply_text = generate_ai_reply(persona, messages, payload.text)
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
    out["last_metrics"] = {**subs, "overall": overall}
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
        "weights": session.get("scoring_weights", {}),
        "created_at": datetime.now(timezone.utc),
    }

    # Coaching feedback generation (LLM or heuristic)
    try:
        feedback = generate_coaching_feedback(session)
        score_doc["feedback"] = feedback
    except Exception:
        pass

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
def leaderboard(period: Literal["7d", "30d", "all"] = "30d", team: Optional[str] = None):
    if db is None:
        raise HTTPException(500, "Database not available")

    since = None
    if period == "7d":
        since = datetime.now(timezone.utc) - timedelta(days=7)
    elif period == "30d":
        since = datetime.now(timezone.utc) - timedelta(days=30)

    match: Dict[str, Any] = {}
    if since:
        match["created_at"] = {"$gte": since}

    if team:
        # find sellers in team
        emails = [s.get("email") for s in db["seller"].find({"team": team}, {"email": 1})]
        if emails:
            match["seller_email"] = {"$in": emails}
        else:
            return []

    pipeline = [
        {"$match": match},
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


# ------------- Dynamic Weights Config ---------------
@app.post("/api/score-config")
def set_score_config(payload: ScoreConfigPayload, user: AuthUser = Depends(get_current_user)):
    if db is None:
        raise HTTPException(500, "Database not available")

    # simple auth: only manager/admin can set global or team
    if payload.scope in ("global", "team") and user.role not in ("manager", "admin"):
        raise HTTPException(403, "Not authorized")

    doc = {
        "scope": payload.scope,
        "team": payload.team,
        "email": payload.email,
        "weights": payload.weights,
        "updated_at": datetime.now(timezone.utc),
    }
    db["scoreconfig"].update_one(
        {"scope": payload.scope, "team": payload.team, "email": payload.email},
        {"$set": doc},
        upsert=True,
    )
    return {"status": "ok"}


@app.get("/api/score-config")
def get_score_config(team: Optional[str] = None, email: Optional[str] = None):
    if db is None:
        raise HTTPException(500, "Database not available")
    # precedence: user > team > global
    doc = None
    if email:
        doc = db["scoreconfig"].find_one({"scope": "user", "email": email})
    if not doc and team:
        doc = db["scoreconfig"].find_one({"scope": "team", "team": team})
    if not doc:
        doc = db["scoreconfig"].find_one({"scope": "global"})
    return to_str_id(doc) if doc else {"weights": {"rapport": 0.25, "discovery": 0.25, "objection": 0.3, "closing": 0.2}}


def get_weight_config_for_user(email: str) -> Optional[Dict[str, float]]:
    # try user, then team (requires seller profile), then global
    seller = db["seller"].find_one({"email": email}) if db is not None else None
    if db is not None:
        doc = db["scoreconfig"].find_one({"scope": "user", "email": email})
        if doc:
            return doc.get("weights")
        team = seller.get("team") if seller else None
        if team:
            doc = db["scoreconfig"].find_one({"scope": "team", "team": team})
            if doc:
                return doc.get("weights")
        doc = db["scoreconfig"].find_one({"scope": "global"})
        if doc:
            return doc.get("weights")
    return None


# ------------- Premium Unlock (Trilhas e Metas) ---------------
@app.get("/api/premium-status")
def premium_status(seller_email: str, last_n: int = 5, threshold: float = 80.0):
    if db is None:
        raise HTTPException(500, "Database not available")
    cur = db["sessionscore"].find({"seller_email": seller_email}).sort("created_at", -1).limit(last_n)
    scores = [d.get("final_score", 0.0) for d in cur]
    if len(scores) < last_n:
        return {"eligible": False, "reason": f"Complete {last_n - len(scores)} more sessions", "last_n": last_n}
    avg = sum(scores) / len(scores)
    return {"eligible": avg >= threshold, "average": round(avg, 1), "last_n": last_n}


# ------------- Coaching Feedback ---------------

def generate_coaching_feedback(session: Dict[str, Any]) -> str:
    # Use LLM if available, else heuristic summary
    history = session.get("messages", [])
    persona_key = session.get("persona_key")
    persona = db["persona"].find_one({"key": persona_key}) or next(
        (p for p in DEFAULT_PERSONAS if p["key"] == persona_key), {}
    )
    if openai_client is None:
        # heuristic: very simple tips
        return (
            "Feedback: Comece construindo rapport (cumprimento, empatia), faça 1-2 perguntas de descoberta (SPIN/BANT),"
            " trate objeções conectando valor e finalize com próximo passo claro."
        )

    system = (
        "Você é um coach sênior de vendas imobiliárias. Dê feedback direto e acionável em 5 bullets:"
        " 1) Rapport, 2) Descoberta (SPIN/BANT), 3) Tratativa de Objeções, 4) Fechamento, 5) Próximos exercícios."
        " Foque em exemplos do diálogo e seja curto (máx 80 palavras)."
    )

    transcript = "\n".join([f"{m['role']}: {m['text']}" for m in history[-20:]])
    persona_desc = f"Persona: {persona.get('name')} ({persona.get('disc_profile')}) | Traços: {', '.join(persona.get('traits', []))} | Gatilhos: {', '.join(persona.get('triggers', []))}"

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{persona_desc}\n\nDiálogo:\n{transcript}"},
    ]

    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.5,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return (
            "Feedback: Mantenha rapport, aprofunde necessidades (SPIN/BANT), conecte valor às objeções e feche pedindo o próximo passo."
        )


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
        "llm": "❌ Disabled",
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

    response["llm"] = "✅ Enabled" if OPENAI_API_KEY else "❌ Disabled"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
