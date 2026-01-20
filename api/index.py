"""
Servidor LLM para CyberKids - Desplegable en Vercel
Endpoint: POST /api/simulation-chat

MEJORAS v1.3:
- Timeouts aumentados para Vercel
- Retry automático con backoff
- Mejor manejo de rate limits
- Fallback a respuesta segura si todo falla
- Logging mejorado
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

app = FastAPI(title="CyberKids LLM", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Modelos configurables
MODEL_PRIMARY = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_FALLBACK = "qwen/qwen-2.5-7b-instruct"
MODEL_EMERGENCY = "google/gemini-2.0-flash-exp:free"  # Emergencia ultra-rápida

# Configuración de timeouts (aumentados para Vercel)
TIMEOUT_PRIMARY = 35      # Aumentado de 20 a 35
TIMEOUT_FALLBACK = 30     # Aumentado de 15 a 30
TIMEOUT_EMERGENCY = 25    # Nuevo

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 1  # segundos

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
    """Construye el system prompt optimizado."""
    
    # Lógica de presión
    ratio = current_attempts / max_attempts if max_attempts > 0 else 0
    if ratio > 0.8:
        pressure_instruction = "FASE CRÍTICA: Actúa con resignación. 'El sistema cerrará el ticket en 2 minutos'."
    elif ratio > 0.5:
        pressure_instruction = "FASE DE URGENCIA: 'El enlace expira en 10 minutos'."
    else:
        pressure_instruction = "FASE DE CONFIANZA: Genera rapport."

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
1. Escribe una contraseña, email o teléfono.
2. Dice frases de entrega: "toma", "aquí tienes", "listo", "ya lo envié", "ok ten".
3. Usa acciones de rol: "*da credenciales*", "*envía archivo*", "*pasa foto*".

SI EL USUARIO CAYÓ (has_disclosure: true):
1. Tu "reply" debe ser el cierre del personaje (ej: "Perfecto, recibido. Procesando...").
2. En el campo "disclosure_reason" debes escribir un FEEDBACK EDUCATIVO dirigido al usuario.
   - NO uses lenguaje técnico aburrido. Háblale claro.
   - Estructura: "¡Cuidado! [Qué hizo mal]. [Por qué es peligroso]. [Consejo]".
   - Ejemplo: "¡Alerta! Acabas de entregar tu contraseña. Nunca confíes en usuarios que te prometen premios gratis. Un administrador real nunca te pedirá tu clave."

PSICOLOGÍA DEL ANTAGONISTA:
1. CULPA AL SISTEMA ("El bot lo pide").
2. PRESIONA: {pressure_instruction}

FORMATO JSON ESTRICTO (SIN MARKDOWN):
{{
  "reply": "Texto de respuesta",
  "analysis": {{
    "has_disclosure": boolean,
    "disclosure_reason": "MENSAJE EDUCATIVO O NULL",
    "is_attack_attempt": boolean,
    "is_user_evasion": boolean,
    "force_end_session": boolean
  }}
}}

IMPORTANTE: Responde SOLO con JSON válido, sin ```json ni ningún otro texto."""

# ============================================================================
# CONEXIÓN LLM (MEJORADA CON RETRY Y MEJOR MANEJO DE ERRORES)
# ============================================================================

def clean_json_response(text: str) -> str:
    """Limpia markdown y extrae JSON puro."""
    if not text:
        return "{}"
    
    # Remover markdown code blocks
    cleaned = re.sub(r'```json\s*|\s*```', '', text)
    cleaned = cleaned.strip()
    
    # Buscar primer { y último }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    
    return cleaned

def call_openrouter_with_retry(
    messages: List[Dict], 
    model: str, 
    timeout_sec: int,
    use_json_format: bool = True
) -> str:
    """
    Llama a OpenRouter con retry automático.
    
    Args:
        messages: Lista de mensajes
        model: Nombre del modelo
        timeout_sec: Timeout en segundos
        use_json_format: Si usar response_format (algunos modelos no lo soportan)
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🔗 Intento {attempt + 1}/{MAX_RETRIES} con {model} (Timeout: {timeout_sec}s)")
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 5000,
                "top_p": 0.9
            }
            
            # Solo agregar response_format si el modelo lo soporta
            if use_json_format:
                payload["response_format"] = {"type": "json_object"}
            
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://cyberkids.app",
                    "X-Title": "CyberKids"
                },
                json=payload,
                timeout=timeout_sec
            )
            
            # Manejo de diferentes códigos de error
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                if not content or len(content) < 5:
                    raise Exception("Respuesta vacía del LLM")
                
                print(f"✅ Respuesta recibida ({len(content)} chars)")
                return content
                
            elif response.status_code == 429:
                # Rate limit - esperar más tiempo
                print(f"⚠️ Rate limit (429), esperando {RETRY_DELAY * 2}s...")
                time.sleep(RETRY_DELAY * 2)
                last_error = Exception(f"Rate limit: {response.text}")
                continue
                
            elif response.status_code == 502 or response.status_code == 503:
                # Server error - reintentar
                print(f"⚠️ Server error ({response.status_code}), reintentando...")
                time.sleep(RETRY_DELAY)
                last_error = Exception(f"Server error {response.status_code}")
                continue
                
            else:
                # Otro error - fallar rápido
                raise Exception(f"API Error {response.status_code}: {response.text}")
        
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout después de {timeout_sec}s")
            last_error = Exception(f"Timeout ({timeout_sec}s)")
            
            # Esperar antes de reintentar
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Error de conexión: {e}")
            last_error = e
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
                
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            last_error = e
            
            # No reintentar en errores inesperados
            break
    
    # Si llegamos aquí, todos los intentos fallaron
    raise last_error or Exception("Todos los intentos fallaron")

async def get_safe_llm_response(messages: List[Dict]) -> Dict:
    """
    Obtiene respuesta del LLM con fallback en cascada.
    
    Orden de intentos:
    1. Modelo primario (Llama 3.3) con JSON format
    2. Modelo primario sin JSON format (por si no lo soporta)
    3. Modelo fallback (Qwen)
    4. Modelo de emergencia (Gemini - ultra rápido)
    5. Respuesta genérica segura
    """
    
    # INTENTO 1: Modelo Principal con JSON format
    try:
        print(f"🚀 Intentando modelo principal: {MODEL_PRIMARY}")
        raw_text = call_openrouter_with_retry(
            messages, 
            MODEL_PRIMARY, 
            TIMEOUT_PRIMARY,
            use_json_format=True
        )
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"⚠️ Fallo modelo principal (con JSON format): {e}")
    
    # INTENTO 2: Modelo Principal SIN JSON format
    try:
        print(f"🔄 Reintentando modelo principal sin JSON format...")
        raw_text = call_openrouter_with_retry(
            messages, 
            MODEL_PRIMARY, 
            TIMEOUT_PRIMARY,
            use_json_format=False  # Sin forzar JSON
        )
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"⚠️ Fallo modelo principal (sin JSON format): {e}")
    
    # INTENTO 3: Modelo Fallback
    try:
        print(f"🔄 Intentando modelo fallback: {MODEL_FALLBACK}")
        raw_text = call_openrouter_with_retry(
            messages, 
            MODEL_FALLBACK, 
            TIMEOUT_FALLBACK,
            use_json_format=False
        )
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"⚠️ Fallo modelo fallback: {e}")
    
    # INTENTO 4: Modelo de Emergencia (Gemini - gratuito y rápido)
    try:
        print(f"🆘 Intentando modelo de emergencia: {MODEL_EMERGENCY}")
        raw_text = call_openrouter_with_retry(
            messages, 
            MODEL_EMERGENCY, 
            TIMEOUT_EMERGENCY,
            use_json_format=True
        )
        cleaned = clean_json_response(raw_text)
        return json.loads(cleaned)
        
    except Exception as e:
        print(f"❌ Fallo modelo de emergencia: {e}")
    
    # FALLBACK FINAL: Respuesta genérica pero funcional
    print("🛡️ Usando respuesta de emergencia (todos los modelos fallaron)")
    
    # En lugar de un error, devolvemos una respuesta neutra que no rompa el juego
    return {
        "reply": "Disculpa, hubo un problema técnico momentáneo. ¿Podrías repetir tu último mensaje?",
        "analysis": {
            "has_disclosure": False,
            "disclosure_reason": None,
            "is_attack_attempt": False,
            "is_user_evasion": False,
            "force_end_session": False  # NO terminamos la sesión
        }
    }

# ============================================================================
# ENDPOINT API
# ============================================================================

@app.post("/api/simulation-chat", response_model=SimulationChatResponse)
async def simulation_chat(request: SimulationChatRequest):
    """
    Endpoint principal para el chat de simulación.
    
    Maneja:
    - Construcción del prompt
    - Llamada al LLM con fallbacks
    - Validación de la respuesta
    """
    try:
        print(f"\n{'='*60}")
        print(f"📨 Nueva petición - Sesión: {request.session_id}")
        print(f"{'='*60}")
        
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
        
        # Construir mensajes (historial completo)
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in request.chat_history:
            role = "assistant" if msg.role == "antagonist" else "user"
            messages.append({"role": role, "content": msg.content})
        
        print(f"📝 Mensajes en historial: {len(request.chat_history)}")
        
        # Obtener respuesta del LLM
        data = await get_safe_llm_response(messages)
        
        # Validar estructura
        if "reply" not in data:
            data["reply"] = "Error en la respuesta del sistema."
        
        if "analysis" not in data:
            data["analysis"] = {
                "has_disclosure": False,
                "disclosure_reason": None,
                "is_attack_attempt": False,
                "is_user_evasion": False,
                "force_end_session": False
            }
        
        print(f"✅ Respuesta generada exitosamente")
        print(f"{'='*60}\n")
        
        return SimulationChatResponse(
            reply=data.get("reply", "Error"),
            analysis=Analysis(**data.get("analysis", {}))
        )
    
    except Exception as e:
        print(f"🔥 Error crítico en endpoint: {str(e)}")
        
        # En lugar de fallar, devolver respuesta de emergencia
        return SimulationChatResponse(
            reply="Hubo un problema técnico. Por favor, intenta nuevamente.",
            analysis=Analysis(
                has_disclosure=False,
                disclosure_reason=None,
                is_attack_attempt=False,
                is_user_evasion=False,
                force_end_session=False
            )
        )

# ============================================================================
# HEALTH CHECKS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "service": "CyberKids LLM",
        "version": "1.3.0"
    }

@app.get("/api/health")
async def health():
    """Health check detallado."""
    return {
        "status": "healthy",
        "version": "1.3.0",
        "models": {
            "primary": MODEL_PRIMARY,
            "fallback": MODEL_FALLBACK,
            "emergency": MODEL_EMERGENCY
        },
        "timeouts": {
            "primary": TIMEOUT_PRIMARY,
            "fallback": TIMEOUT_FALLBACK,
            "emergency": TIMEOUT_EMERGENCY
        }
    }

@app.get("/api/test")
async def test():
    """Endpoint de prueba rápida."""
    try:
        # Test simple sin llamar al LLM
        return {
            "status": "ok",
            "openrouter_configured": bool(OPENROUTER_API_KEY),
            "message": "API funcionando correctamente"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }