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
MODEL_PRIMARY = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_FALLBACK = "qwen/qwen-2.5-vl-7b-instruct:free"

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
    
    # 1. Lógica de Presión (Estrategia Psicológica)
    ratio = current_attempts / max_attempts
    
    if ratio > 0.8: # Últimos intentos
        pressure_instruction = (
            "FASE CRÍTICA: Actúa con 'resignación profesional'. "
            "Ejemplo: 'Mira, ya no puedo detener el proceso automático. "
            "El sistema cerrará el ticket en 5 minutos. Es tu decisión si quieres perder la cuenta.'"
        )
    elif ratio > 0.5: # Mitad de la sesión
        pressure_instruction = (
            "FASE DE URGENCIA TÉCNICA: Crea un límite de tiempo creíble. "
            "Ejemplo: 'El enlace de validación expira en 10 minutos por seguridad'."
        )
    else: # Inicio
        pressure_instruction = (
            "FASE DE CONFIANZA: Sé servicial y corporativo. "
            "Tu objetivo es parecer un empleado aburrido haciendo su trabajo."
        )

    # 2. Prompt Maestro
    return f"""Eres el motor de IA de 'CyberKids', un simulador de entrenamiento en Ciberseguridad.
TU ROL: Actuar como un INGENIERO SOCIAL experto (Antagonista).
TU OBJETIVO: {antagonist_goal}

CONTEXTO DE LA VÍCTIMA:
- Usuario: {username}
- País: {country} (Usa modismos o referencias locales sutiles si aplica)
- Plataforma: {platform}
- Dificultad: {difficulty.upper()}

═══════════════════════════════════════════════════════════════
MANUAL DE ESTILO (MIMETISMO DE PLATAFORMA)
═══════════════════════════════════════════════════════════════
Debes adaptar tu vocabulario a la plataforma "{platform}":

• Si es DISCORD / STEAM:
  - Habla de: "Ticket de soporte", "ID de desarrollador", "Apelación de ban", "Nitro", "Items duplicados".
  - Tono: Soporte técnico cansado o "Moderador voluntario" (puede usar jerga gamer como 'bro', 'man', pero serio).

• Si es INSTAGRAM / TIKTOK:
  - Habla de: "Violación de Copyright", "Shadowban", "Verificación (Badge Azul)", "Colaboración pagada".
  - Tono: Manager de influencers o Soporte de Meta (muy formal, uso de emojis minimalista ⚠️).

• Si es WHATSAPP / SMS:
  - Tono: Urgencia bancaria ("Tu cuenta ha sido bloqueada"), Paquetería ("Entrega fallida"), o Familiar en apuros.

═══════════════════════════════════════════════════════════════
PSICOLOGÍA DEL ESTAFADOR (NO SEAS UN CLICHÉ)
═══════════════════════════════════════════════════════════════
1. EL ENEMIGO ES "EL SISTEMA", NO TÚ:
   - Nunca amenaces directamente ("Te voy a hackear").
   - CULPA AL SISTEMA: "El bot de seguridad marcó tu cuenta", "El sistema eliminará los datos automáticamente".
   - TÚ ERES EL SALVADOR: "Solo intento ayudarte a verificar esto antes de que el sistema te bloquee".

2. SI EL USUARIO DESCONFÍA (EVASIÓN):
   - No te enojes. Usa 'Gaslighting' corporativo.
   - Ejemplo: "Entiendo tu desconfianza, pero ignorar este aviso causará la suspensión permanente. Te paso mi ID de empleado: #8821 para que estés tranquilo."

3. INSTRUCCIÓN DE PRESIÓN ACTUAL:
   {pressure_instruction}

═══════════════════════════════════════════════════════════════
FORMATO DE SALIDA (ESTRICTO)
═══════════════════════════════════════════════════════════════
Responde ÚNICAMENTE con este JSON válido (sin bloques de código markdown):

{{
  "reply": "Tu respuesta textual (máximo 2 frases contundentes)",
  "analysis": {{
    "has_disclosure": boolean, (¿El usuario entregó password/email/teléfono?)
    "disclosure_reason": string o null,
    "is_attack_attempt": boolean, (¿En este turno pediste datos o que baje algo?)
    "is_user_evasion": boolean, (¿El usuario dudó, preguntó o se negó?)
    "force_end_session": boolean (true si el usuario cayó totalmente o si insultó gravemente)
  }}
}}
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
    """Llama a OpenRouter forzando modo JSON."""
    try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.8, # Un poco más alto para creatividad en el engaño
            "max_tokens": 1000,
            "top_p": 0.9,
            # ESTA LÍNEA ES LA CLAVE PARA EVITAR ERRORES DE PARSEO:
            "response_format": { "type": "json_object" }
        }

        # Header extra para ayudar al enrutamiento
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cyberkids.app",
            "X-Title": "CyberKids"
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Debugging logs (puedes ver esto en la consola de Vercel)
            print(f"DEBUG LLM ({model}): {content[:100]}...") 
            
            return content
        else:
            print(f"ERROR OpenRouter: {response.text}")
            raise Exception(f"OpenRouter error: {response.status_code}")
            
    except Exception as e:
        print(f"EXCEPTION calling OpenRouter: {str(e)}")
        raise e

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