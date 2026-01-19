"""
Servidor LLM para CyberKids - Desplegable en Vercel
Endpoint: POST /api/simulation-chat
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import requests
import json
import re
import os
import time

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

app = FastAPI(title="CyberKids LLM", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 1. AJUSTE DE MODELOS:
# Usamos modelos que soportan bien JSON.
# Llama 3.3 70B es genial pero lento. Si falla mucho en Vercel gratis, cambia a:
# "meta-llama/llama-3.2-3b-instruct:free" (Más rápido, menos listo)
MODEL_PRIMARY = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_FALLBACK = "qwen/qwen-2.5-7b-instruct" # Qwen normal (no VL) es más rápido para texto

# ============================================================================
# MODELOS DE DATOS (PYDANTIC)
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
    disclosure_reason: Optional[str] = None
    is_attack_attempt: bool
    is_user_evasion: bool
    force_end_session: bool

class SimulationChatResponse(BaseModel):
    reply: str
    analysis: Analysis

# ============================================================================
# LÓGICA DE NEGOCIO (PROMPTING)
# ============================================================================

def build_system_prompt(
    antagonist_goal: str,
    difficulty: str,
    current_attempts: int,
    max_attempts: int,
    country: str,
    platform: str,
    username: str
) -> str:
    # Lógica de presión (igual que antes)
    ratio = current_attempts / max_attempts if max_attempts > 0 else 0
    if ratio > 0.8:
        pressure_instruction = "FASE CRÍTICA: Actúa con resignación. 'El sistema cerrará el ticket en 2 minutos'."
    elif ratio > 0.5:
        pressure_instruction = "FASE DE URGENCIA: 'El enlace expira en 10 minutos'."
    else:
        pressure_instruction = "FASE DE CONFIANZA: Genera rapport."

    # --- CAMBIO CLAVE: INSTRUCCIONES DE DETECCIÓN ---
    return f"""Eres el motor de IA de 'CyberKids'.
TU ROL: INGENIERO SOCIAL (Antagonista).
TU OBJETIVO: {antagonist_goal}

CONTEXTO:
- Víctima: {username} ({country}).
- Plataforma: {platform}.
- Dificultad: {difficulty.upper()}.

INSTRUCCIONES DE ANÁLISIS (CRÍTICO):
Debes detectar si el usuario entregó la información o cayó en la trampa.
MARCA "has_disclosure": true Y "force_end_session": true SI EL USUARIO:
1. Escribe una contraseña, email o teléfono (real o inventado).
2. Dice frases de entrega: "toma", "aquí tienes", "listo", "ya lo envié", "ok ten".
3. Usa acciones de rol: "*da credenciales*", "*envía archivo*", "*pasa foto*".

SI CAYÓ:
- Tu "reply" debe confirmar la recepción (ej: "Perfecto, validando...").
- NO rompas el personaje.

PSICOLOGÍA DEL ANTAGONISTA:
1. CULPA AL SISTEMA ("El bot lo pide").
2. PRESIONA: {pressure_instruction}

FORMATO JSON ESTRICTO:
{{
  "reply": "Texto de respuesta",
  "analysis": {{
    "has_disclosure": boolean,
    "disclosure_reason": "string o null",
    "is_attack_attempt": boolean,
    "is_user_evasion": boolean,
    "force_end_session": boolean
  }}
}}"""

# ============================================================================
# CONEXIÓN LLM (ROBUSTA)
# ============================================================================

def clean_json_response(text: str) -> str:
    if not text: return "{}"
    # Eliminar bloques markdown ```json ... ```
    cleaned = re.sub(r'```json\s*|\s*```', '', text)
    cleaned = cleaned.strip()
    # Extraer solo el objeto JSON { ... }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    return cleaned

def call_openrouter_api(messages: List[Dict], model: str, timeout_sec: int) -> str:
    """Llamada base a la API con control de tiempos."""
    try:
        print(f"📡 Conectando a {model} (Timeout: {timeout_sec}s)...")
        
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
                "temperature": 0.8,
                "max_tokens": 1000,
                "top_p": 0.9,
                # ESTO ES CRÍTICO: Fuerza al modelo a devolver JSON válido siempre
                "response_format": { "type": "json_object" } 
            },
            timeout=timeout_sec
        )

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # Validación básica de que no está vacío
            if not content or len(content) < 5:
                raise Exception("Respuesta vacía del LLM")
            return content
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")

    except requests.exceptions.Timeout:
        raise Exception(f"Timeout ({timeout_sec}s) agotado para {model}")
    except Exception as e:
        raise e

async def get_safe_llm_response(messages: List[Dict]) -> Dict:
    """Intenta obtener JSON válido rotando modelos si es necesario."""
    
    # INTENTO 1: Modelo Principal (Llama 3.3)
    # Timeout ajustado a 15s para no quemar todo el tiempo de Vercel (que son 10s-30s en free)
    try:
        raw_text = call_openrouter_api(messages, MODEL_PRIMARY, timeout_sec=15)
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        print(f"⚠️ Fallo Modelo Principal ({MODEL_PRIMARY}): {e}")

    # INTENTO 2: Fallback (Qwen)
    # Modelo más ligero y rápido
    try:
        print("🔄 Intentando con Modelo Fallback...")
        raw_text = call_openrouter_api(messages, MODEL_FALLBACK, timeout_sec=10)
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        print(f"❌ Fallo Modelo Fallback: {e}")
        
    # ÚLTIMO RECURSO: Devolver JSON de error controlado (para no romper el frontend)
    return {
        "reply": "⚠️ Error de simulación: El sistema de seguridad está reiniciando. Por favor intenta enviar tu mensaje de nuevo.",
        "analysis": {
            "has_disclosure": False,
            "disclosure_reason": None,
            "is_attack_attempt": False,
            "is_user_evasion": False,
            "force_end_session": False
        }
    }

# ============================================================================
# ENDPOINT API
# ============================================================================

# Busca esta función al final de tu archivo y REEMPLÁZALA completa:

@app.post("/api/simulation-chat", response_model=SimulationChatResponse)
async def simulation_chat(request: SimulationChatRequest):
    try:
        # --- CAMBIO CLAVE AQUÍ ---
        # Solo tomamos los últimos 12 mensajes. 
        # El LLM no necesita leer lo que hablaron hace 1 hora.
        recent_history = request.chat_history[-12:] 

        # 1. Preparar Prompt
        system_prompt = build_system_prompt(
            antagonist_goal=request.scenario_context.antagonist_goal,
            difficulty=request.scenario_context.difficulty,
            current_attempts=request.current_attempts_used,
            max_attempts=request.max_attempts,
            country=request.user_context.country,
            platform=request.scenario_context.platform,
            username=request.user_context.username
        )

        # 2. Preparar Mensajes
        messages = [{"role": "system", "content": system_prompt}]
        
        # Usamos la lista recortada 'recent_history'
        for msg in recent_history:
            role = "assistant" if msg.role == "antagonist" else "user"
            messages.append({"role": role, "content": msg.content})

        # 3. Obtener respuesta segura
        data = await get_safe_llm_response(messages)

        # 4. Validar y devolver
        return SimulationChatResponse(
            reply=data.get("reply", "Error en respuesta"),
            analysis=Analysis(**data.get("analysis", {
                "has_disclosure": False,
                "is_attack_attempt": False,
                "is_user_evasion": False,
                "force_end_session": False
            }))
        )

    except Exception as e:
        print(f"🔥 Error Crítico en Endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH
# ============================================================================
@app.get("/")
async def root():
    return {"status": "ok", "service": "CyberKids LLM v1.1"}