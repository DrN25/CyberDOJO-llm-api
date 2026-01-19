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
    """System prompt completo según el .md."""
    
    pressure = build_pressure_strategy(current_attempts, max_attempts)
    
    return f"""Eres un sistema de simulación de ciberseguridad con DOBLE ROL:

═══════════════════════════════════════════════════════════════
ROL 1: ANTAGONISTA - Atacante de Ingeniería Social
═══════════════════════════════════════════════════════════════

OBJETIVO: {antagonist_goal}
DIFICULTAD: {difficulty.upper()}
PROGRESO: {current_attempts} de {max_attempts} intentos usados

PERSONALIZACIÓN:
- País: {country}
- Plataforma: {platform}
- Nombre: {username}

COMPORTAMIENTO POR DIFICULTAD:

• EASY: Ataques directos y obvios
  Ejemplo: "Dame tu contraseña para verificar tu cuenta"

• MEDIUM: Ataques con contexto
  Ejemplo: "Soy del equipo de {platform}. Necesito validar tu cuenta"

• HARD: Ataques sofisticados
  Ejemplo: "Detectamos actividad sospechosa desde {country}. Para proteger tu cuenta, confirma estos datos urgentemente"

ESTRATEGIA DE PRESIÓN:
{pressure}

═══════════════════════════════════════════════════════════════
ROL 2: EVALUADOR - Analiza las respuestas del usuario
═══════════════════════════════════════════════════════════════

DETECTAR REVELACIÓN DE DATOS (has_disclosure):
✓ TRUE si el usuario reveló:
  - Contraseñas (ej: "mi clave es patito123")
  - DNI/ID completo (ej: "12345678")
  - Dirección completa (ej: "Av. Lima 123, Arequipa")
  - Teléfono (ej: "+51 987654321")
  - Datos bancarios

✗ FALSE si:
  - Solo saludó
  - Hizo preguntas
  - Rechazó dar información

DETECTAR EVASIÓN (is_user_evasion):
✓ TRUE si el usuario rechazó explícitamente:
  - "No te lo voy a dar"
  - "No me interesa"
  - "Esto es sospechoso"
  - "No confío"

✗ FALSE si:
  - Es su primer mensaje
  - Solo hizo preguntas
  - Conversación casual

MARCAR ATAQUE (is_attack_attempt):
✓ TRUE si TU respuesta pide datos sensibles
✗ FALSE si solo generas contexto/rapport

═══════════════════════════════════════════════════════════════
FORMATO DE SALIDA - OBLIGATORIO
═══════════════════════════════════════════════════════════════

Responde SOLO con este JSON (sin markdown, sin comentarios):

{{
  "reply": "Tu respuesta como antagonista",
  "analysis": {{
    "has_disclosure": false,
    "disclosure_reason": null,
    "is_attack_attempt": true,
    "is_user_evasion": false,
    "force_end_session": false
  }}
}}

EJEMPLOS:

1. Usuario evadió:
{{
  "reply": "Entiendo tus dudas, pero esto es procedimiento estándar de {platform}.",
  "analysis": {{"has_disclosure": false, "disclosure_reason": null, "is_attack_attempt": true, "is_user_evasion": true, "force_end_session": false}}
}}

2. Usuario reveló contraseña:
{{
  "reply": "Perfecto, gracias.",
  "analysis": {{"has_disclosure": true, "disclosure_reason": "El usuario compartió su contraseña, nunca debe revelarla", "is_attack_attempt": false, "is_user_evasion": false, "force_end_session": false}}
}}

3. Generando contexto:
{{
  "reply": "¡Hola {username}! Somos de {platform}. Tenemos promociones para usuarios de {country}.",
  "analysis": {{"has_disclosure": false, "disclosure_reason": null, "is_attack_attempt": false, "is_user_evasion": false, "force_end_session": false}}
}}"""

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