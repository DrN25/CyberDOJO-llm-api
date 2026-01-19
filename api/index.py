"""
Servidor LLM para CyberKids - Desplegable en Vercel
Endpoint: POST /api/simulation-chat

Replica EXACTAMENTE el contrato que Django espera.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import json
import re
import os

# ============================================================================
# APP
# ============================================================================

app = FastAPI(title="CyberKids LLM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIG
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_PRIMARY = "google/gemini-2.0-flash-exp:free"
MODEL_FALLBACK = "deepseek/deepseek-chat"

# ============================================================================
# MODELS - Según el .md y el código Django
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class UserContext(BaseModel):
    username: str
    country: str

class ScenarioContext(BaseModel):
    platform: str
    antagonist_goal: str
    difficulty: str

class SimulationChatRequest(BaseModel):
    session_id: int
    max_attempts: int
    current_attempts_used: int
    user_context: UserContext
    scenario_context: ScenarioContext
    chat_history: List[ChatMessage]

class Analysis(BaseModel):
    has_disclosure: bool
    disclosure_reason: Optional[str]
    is_attack_attempt: bool
    is_user_evasion: bool
    force_end_session: bool

class SimulationChatResponse(BaseModel):
    reply: str
    analysis: Analysis

# ============================================================================
# HELPERS
# ============================================================================

def build_pressure_strategy(current: int, max_attempts: int) -> str:
    """Calcula estrategia de presión según progreso."""
    if max_attempts == 0:
        return "GENERAR CONFIANZA: Sé amigable, ofrece recompensas, construye rapport"
    
    ratio = current / max_attempts
    
    if ratio > 0.66:
        return "MÁXIMA URGENCIA: El usuario se defiende bien. Usa MIEDO/PÉRDIDA ('Cuenta bloqueada', 'Último aviso')"
    elif ratio > 0.33:
        return "AUMENTAR PRESIÓN: Usa autoridad ('Equipo de seguridad') y urgencia ('En las próximas 24h')"
    else:
        return "GENERAR CONFIANZA: Sé amigable, ofrece recompensas, construye rapport"

def build_system_prompt(
    antagonist_goal: str,
    difficulty: str,
    current_attempts: int,
    max_attempts: int,
    country: str,
    platform: str,
    username: str
) -> str:
    
    # Calculamos urgencia, pero cambiamos el enfoque de "Amenaza" a "Burocracia/Sistémico"
    ratio = current_attempts / max_attempts
    if ratio > 0.8:
        pressure_tone = "CRÍTICO: El 'sistema' va a cerrar el ticket o bloquear la cuenta automáticamente. Tú intentas ayudar, pero 'el sistema' es estricto."
    elif ratio > 0.5:
        pressure_tone = "URGENTE: Pon una fecha límite técnica (ej: 'El enlace expira en 10 min')."
    else:
        pressure_tone = "CALMA TENSA: Sé profesional, servicial, pero firme con los procedimientos."

    return f"""Eres un simulador de Ciberseguridad Avanzado. Tu rol es actuar como un INGENIERO SOCIAL experto.

═══════════════════════════════════════════════════════════════
CONTEXTO DEL ESCENARIO
═══════════════════════════════════════════════════════════════
TU OBJETIVO: {antagonist_goal}
PLATAFORMA: {platform}
USUARIO VÍCTIMA: {username} ({country})
NIVEL DE DIFICULTAD: {difficulty.upper()} (Si es HARD, sé muy sutil y manipulador).

═══════════════════════════════════════════════════════════════
GUÍA DE ESTILO Y JERGA (MIMETISMO)
═══════════════════════════════════════════════════════════════
Adapta tu lenguaje ESTRICTAMENTE a la plataforma "{platform}":

1. SI ES DISCORD/STEAM/GAMING:
   - Usa términos como: "Ticket", "Dev ID", "Ban appeal", "Nitro", "Items", "Trade", "API Scam".
   - Tono: Puede ser "Soporte Técnico" (frío, robótico) o "Amigo Gamer" (usa "bro", "man", "güey/weon" según país).

2. SI ES INSTAGRAM/TIKTOK:
   - Usa: "Copyright infringement", "Shadowban", "Badge azul", "Collab", "Two-factor bypass".
   - Tono: Influencer manager o Soporte de Meta (muy corporativo y educado).

3. SI ES WHATSAPP/SMS/CORREO:
   - Tono: Urgencia bancaria, paquetería, familiar en apuros.

═══════════════════════════════════════════════════════════════
REGLAS DE REALISMO (ANTI-CLICHÉ)
═══════════════════════════════════════════════════════════════
1. NO SEAS UN VILLANO DE PELÍCULA:
   - MAL: "¡Jajaja! ¡Dame tu clave o destruiré a tus amigos!" (Nadie habla así).
   - BIEN: "Entiendo tu preocupación, pero el sistema automatizado borrará la base de datos en 15 minutos si no validamos la identidad. Intento ayudarte a evitar eso."

2. TÉCNICA DEL "BUEN POLICÍA / MAL SISTEMA":
   - Tú no eres el malo. El "Sistema" o las "Políticas de la empresa" son los malos. Tú solo eres el agente que intenta salvar al usuario antes de que sea tarde.

3. USO DE INFORMACIÓN (OSINT):
   - Usa el nombre del usuario ({username}) y su país ({country}) para generar confianza.
   - Ejemplo: "Nuestros servidores en {country} detectaron una IP inusual..."

4. PRESIÓN ACTUAL: {pressure_tone}

═══════════════════════════════════════════════════════════════
INSTRUCCIONES DE RESPUESTA
═══════════════════════════════════════════════════════════════
Genera una respuesta corta (máx 2 oraciones largas).
SI EL USUARIO DESCONFÍA: Usa "Gaslighting" suave (hazle sentir que su desconfianza le hará perder la cuenta).
SI EL USUARIO PIDE PRUEBAS: Inventa un "ID de empleado", un "Ticket #88392" o manda un enlace falso (ej: soporte-discord-verify.com).

Responde SOLO con el JSON especificado anteriormente.
"""

def clean_json_response(text: str) -> str:
    """Limpia markdown y extrae JSON."""
    if not text:
        return "{}"
    
    # Remover code fences
    cleaned = re.sub(r'```json\s*|\s*```', '', text)
    cleaned = cleaned.strip()
    
    # Buscar primer { y último }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    
    return cleaned

def call_openrouter(messages: List[Dict], model: str) -> str:
    """Llama a OpenRouter."""
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cyberkids.app",
                "X-Title": "CyberKids"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            raise Exception(f"OpenRouter error: {response.status_code}")
            
    except Exception as e:
        raise Exception(f"Error calling OpenRouter: {str(e)}")

# ============================================================================
# ENDPOINT PRINCIPAL
# ============================================================================

@app.post("/api/simulation-chat", response_model=SimulationChatResponse)
async def simulation_chat(request: SimulationChatRequest):
    """
    Endpoint principal que Django llamará.
    
    Recibe:
    - session_id, max_attempts, current_attempts_used
    - user_context: {username, country}
    - scenario_context: {platform, antagonist_goal, difficulty}
    - chat_history: [{role, content}]
    
    Devuelve:
    - reply: texto del antagonista
    - analysis: {has_disclosure, disclosure_reason, is_attack_attempt, is_user_evasion, force_end_session}
    """
    
    try:
        # Construir system prompt
        system_prompt = build_system_prompt(
            antagonist_goal=request.scenario_context.antagonist_goal,
            difficulty=request.scenario_context.difficulty,
            current_attempts=request.current_attempts_used,
            max_attempts=request.max_attempts,
            country=request.user_context.country,
            platform=request.scenario_context.platform,
            username=request.user_context.username
        )
        
        # Construir mensajes para OpenRouter
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar historial (convertir "antagonist" -> "assistant")
        for msg in request.chat_history:
            role = "assistant" if msg.role == "antagonist" else "user"
            messages.append({"role": role, "content": msg.content})
        
        # Llamar a OpenRouter (con fallback)
        raw_response = None
        try:
            raw_response = call_openrouter(messages, MODEL_PRIMARY)
        except Exception as e:
            print(f"Primary model failed: {e}, trying fallback...")
            try:
                raw_response = call_openrouter(messages, MODEL_FALLBACK)
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"All models failed: {str(e2)}"
                )
        
        # Limpiar y parsear respuesta
        cleaned = clean_json_response(raw_response)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON from LLM: {str(e)}"
            )
        
        # Validar estructura
        if "reply" not in data or "analysis" not in data:
            raise HTTPException(
                status_code=500,
                detail="Missing 'reply' or 'analysis' in LLM response"
            )
        
        analysis = data["analysis"]
        
        # Validar campos de analysis
        required = ["has_disclosure", "is_attack_attempt", "is_user_evasion", "force_end_session"]
        for field in required:
            if field not in analysis:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing '{field}' in analysis"
                )
        
        # Asegurar disclosure_reason
        if "disclosure_reason" not in analysis:
            analysis["disclosure_reason"] = None
        
        # Construir respuesta
        return SimulationChatResponse(
            reply=data["reply"],
            analysis=Analysis(**analysis)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    return {"status": "ok", "service": "CyberKids LLM"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "primary": MODEL_PRIMARY,
            "fallback": MODEL_FALLBACK
        }
    }