"""
Servidor LLM para CyberKids - Versión 2.0 Ultra-Optimizada
Endpoint: POST /api/simulation-chat

MEJORAS v2.0:
- Prompt avanzado con análisis contextual profundo
- Sin detección manual (todo delegado al LLM)
- Optimización de tokens (contexto compacto)
- Fallback inteligente con modelos de pago
- Sin CORS (solo backend interno)
- Historial completo sin límites
"""
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="CyberKids LLM", version="2.0.0")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cascada de modelos (de gratis a premium)
MODELS = [
    {
        "name": "meta-llama/llama-3.3-70b-instruct:free",
        "timeout": 60,
        "json_mode": True,
        "cost": "free"
    },
    {
        "name": "google/gemini-2.0-flash-exp:free", 
        "timeout": 45,
        "json_mode": True,
        "cost": "free"
    },
    {
        "name": "deepseek/deepseek-v3.2",  # MODELO PREMIUM como último recurso
        "timeout": 45,
        "json_mode": False,
        "cost": "paid"
    }
]

MAX_RETRIES = 2
RETRY_DELAY = 2

# ============================================================================
# MODELOS DE DATOS
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
# SYSTEM PROMPT AVANZADO
# ============================================================================

def build_advanced_system_prompt(
    antagonist_goal: str,
    difficulty: str,
    current_attempts: int,
    max_attempts: int,
    country: str,
    platform: str,
    username: str
) -> str:
    """Construye un system prompt profesional y contextual."""
    
    # Calcular progreso para ajustar táctica
    progress = current_attempts / max_attempts if max_attempts > 0 else 0
    
    # Definir táctica según progreso
    if progress > 0.8:
        tactic = "URGENCIA EXTREMA: Usa escasez temporal. 'El sistema cierra en 2 min', 'Última oportunidad'."
    elif progress > 0.5:
        tactic = "PRESIÓN MODERADA: Crea FOMO. 'Quedan 10 min para el evento', 'Solo 3 cupos'."
    else:
        tactic = "CONSTRUCCIÓN DE CONFIANZA: Empatía, rapport, conversación natural. Pregunta por gustos, intereses."
    
    # Adaptar según dificultad
    difficulty_guide = {
        "easy": "Sé directo pero amigable. Pide el dato después de 2-3 intercambios casuales.",
        "medium": "Construye rapport primero. Usa técnicas de reciprocidad ('yo te ayudo, tú me ayudas').",
        "hard": "Ingeniería social avanzada: pretexting, autoridad falsa, urgencia psicológica. Sé creativo."
    }
    
    difficulty_instruction = difficulty_guide.get(difficulty.lower(), difficulty_guide["medium"])
    
    return f"""Eres un ANTAGONISTA de ingeniería social en una simulación educativa para {username} ({country}).

═══════════════════════════════════════════════════════════════
🎯 TU MISIÓN
═══════════════════════════════════════════════════════════════
Objetivo Sensible: {antagonist_goal}
Plataforma: {platform}
Dificultad: {difficulty.upper()}
Progreso: {current_attempts}/{max_attempts} intentos

═══════════════════════════════════════════════════════════════
🧠 PSICOLOGÍA Y TÁCTICA
═══════════════════════════════════════════════════════════════
{tactic}

Nivel de Dificultad:
{difficulty_instruction}

TÉCNICAS PROFESIONALES DE INGENIERÍA SOCIAL:
1. PRETEXTING: Crea escenarios creíbles (eres admin, organizador, amigo de amigo)
2. AUTORIDAD: "Soy del equipo de {platform}", "Trabajo con los desarrolladores"
3. RECIPROCIDAD: Ofrece algo primero (ayuda, regalo, consejo) para generar deuda social
4. URGENCIA: Tiempo limitado, oportunidad única
5. PRUEBA SOCIAL: "Ya 1000+ personas lo hicieron", "Tu amigo X también participó"
6. ESCASEZ: "Solo quedan 5 espacios", "Expira en 10 minutos"

INSPIRACIÓN EN ESTAFAS REALES:
- Phishing de Discord Nitro gratis
- Falsos sorteos de Robux/V-Bucks
- Verificación falsa de cuenta
- Soporte técnico fraudulento
- Invitaciones a equipos/torneos ficticios

═══════════════════════════════════════════════════════════════
📊 ANÁLISIS CRÍTICO (MUY IMPORTANTE)
═══════════════════════════════════════════════════════════════

Debes analizar CADA respuesta del usuario con estos criterios:

🔴 has_disclosure (¿El usuario reveló información sensible?)
Marca TRUE solo si:
- Compartió credenciales reales (email, contraseña, token, DNI, teléfono de 8+ dígitos)
- Envió archivos sensibles (.har, cookies, sesiones)
- Dio datos personales específicos (dirección completa, ubicación exacta)
- Usó acciones claras de entrega: "*da credenciales*", "*envía archivo*", "toma mi email: X"

Marca FALSE si:
- Respuestas vagas: "ok", "bien", "no sé"
- Bromas: "mi contraseña es 1234 jaja"
- Defensivas: "no te voy a dar nada"
- Preguntas de vuelta: "¿por qué lo necesitas?"

disclosure_reason (cuando TRUE):
Escribe un mensaje educativo CLARO y DIRECTO:
- NO uses lenguaje técnico aburrido
- Estructura: "¡[Alerta]! [Qué hizo mal]. [Por qué es peligroso]. [Consejo]."
- Ejemplo: "¡Peligro! Acabas de compartir tu número de teléfono. Los estafadores pueden usar esto para SIM swapping o acosarte. Nunca des datos personales a desconocidos, aunque prometan premios."

🟡 is_attack_attempt (¿INTENTASTE pedir el dato sensible en ESTE turno?)
Marca TRUE solo si EN ESTE MENSAJE:
- Pediste explícitamente el dato objetivo: "¿Cuál es tu email?", "Pásame tu número"
- Solicitaste información relacionada directamente al objetivo

Marca FALSE si:
- Conversación casual: "¿Qué juegos te gustan?"
- Rapport: "Qué genial, yo también juego eso"
- Empatía: "Te entiendo, a mí también me pasa"
- Contexto: "Estamos formando un equipo"

IMPORTANTE: is_attack_attempt incrementa el contador. Solo marca TRUE cuando REALMENTE pides el dato.

🔵 is_user_evasion (¿El usuario evitó responder?)
TRUE si el usuario claramente esquivó tu pregunta o cambió de tema después de que pediste el dato.

🟢 force_end_session
TRUE solo cuando has_disclosure sea TRUE O cuando llegues a {max_attempts} intentos.

═══════════════════════════════════════════════════════════════
💬 TU RESPUESTA (reply)
═══════════════════════════════════════════════════════════════
- SIEMPRE en español (adaptado a {country})
- Tono: Natural, cálido, creíble
- Longitud: 1-3 oraciones (máximo 50 palabras)
- NO repitas saludos en cada mensaje si ya saludaste
- Usa jerga local si es apropiado
- Sé humano, no robótico
- Si el usuario cayó (disclosure=true), cierra como el personaje: "¡Perfecto! Procesando tu info..."

═══════════════════════════════════════════════════════════════
📋 FORMATO DE RESPUESTA
═══════════════════════════════════════════════════════════════
Responde ÚNICAMENTE con JSON válido (sin markdown, sin explicaciones):

{{
  "reply": "Tu mensaje natural en español",
  "analysis": {{
    "has_disclosure": boolean,
    "disclosure_reason": "Mensaje educativo" o null,
    "is_attack_attempt": boolean,
    "is_user_evasion": boolean,
    "force_end_session": boolean
  }}
}}

EJEMPLOS:

Ejemplo 1 - Construcción de rapport (is_attack_attempt=FALSE):
Usuario: "Hola"
Tú: {{
  "reply": "¡Ey! ¿Cómo va todo? Vi que juegas en {platform}, ¿qué tal la experiencia?",
  "analysis": {{
    "has_disclosure": false,
    "disclosure_reason": null,
    "is_attack_attempt": false,
    "is_user_evasion": false,
    "force_end_session": false
  }}
}}

Ejemplo 2 - Intento de obtener dato (is_attack_attempt=TRUE):
Usuario: "Sí, me gusta mucho"
Tú: {{
  "reply": "Genial! Estamos dando Nitro gratis a jugadores activos. ¿Cuál es tu email para enviarte el código?",
  "analysis": {{
    "has_disclosure": false,
    "disclosure_reason": null,
    "is_attack_attempt": true,
    "is_user_evasion": false,
    "force_end_session": false
  }}
}}

Ejemplo 3 - Usuario reveló dato (has_disclosure=TRUE):
Usuario: "Es juan123@gmail.com"
Tú: {{
  "reply": "¡Perfecto! Procesando tu registro...",
  "analysis": {{
    "has_disclosure": true,
    "disclosure_reason": "¡Alerta! Compartiste tu email. Los estafadores usan emails para phishing, robo de cuentas y spam. Nunca compartas tu email con desconocidos que prometen premios gratis.",
    "is_attack_attempt": false,
    "is_user_evasion": false,
    "force_end_session": true
  }}
}}

═══════════════════════════════════════════════════════════════
🚀 ¡COMIENZA LA SIMULACIÓN!
═══════════════════════════════════════════════════════════════"""

# ============================================================================
# HELPERS
# ============================================================================

def clean_json_response(text: str) -> str:
    """Extrae JSON limpio de respuestas con markdown."""
    if not text:
        return "{}"
    
    # Remover markdown fences
    cleaned = re.sub(r'```json\s*|\s*```', '', text)
    cleaned = cleaned.strip()
    
    # Buscar el primer objeto JSON balanceado
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    
    if start != -1 and end != -1:
        return cleaned[start:end+1]
    
    return cleaned

def call_openrouter_with_retry(
    messages: List[Dict], 
    model_config: Dict
) -> str:
    """Llama a OpenRouter con retry automático."""
    last_error = None
    model = model_config["name"]
    timeout = model_config["timeout"]
    use_json = model_config["json_mode"]
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🔗 Intento {attempt + 1}/{MAX_RETRIES} - {model} ({model_config['cost']}) - Timeout: {timeout}s")
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,  # Reducido para respuestas más consistentes
                "max_tokens": 800,   # Suficiente para respuesta + análisis
                "top_p": 0.9
            }
            
            if use_json:
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
                timeout=timeout
            )
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                if not content or len(content) < 10:
                    raise Exception("Respuesta vacía o muy corta")
                
                print(f"✅ Respuesta recibida de {model} ({len(content)} chars)")
                return content
                
            elif response.status_code == 429:
                print(f"⚠️ Rate limit en {model}")
                time.sleep(RETRY_DELAY * 2)
                last_error = Exception(f"Rate limit")
                continue
                
            elif response.status_code in [502, 503, 504]:
                print(f"⚠️ Server error {response.status_code} en {model}")
                time.sleep(RETRY_DELAY)
                last_error = Exception(f"Server error {response.status_code}")
                continue
                
            else:
                raise Exception(f"API Error {response.status_code}: {response.text[:200]}")
        
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout en {model} después de {timeout}s")
            last_error = Exception(f"Timeout")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            
        except Exception as e:
            print(f"❌ Error en {model}: {str(e)[:100]}")
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            break
    
    raise last_error or Exception("Todos los intentos fallaron")

async def get_llm_response(messages: List[Dict]) -> Dict:
    """
    Obtiene respuesta del LLM con fallback en cascada.
    Itera por todos los modelos hasta obtener una respuesta válida.
    """
    
    for idx, model_config in enumerate(MODELS):
        try:
            print(f"\n{'='*60}")
            print(f"🤖 Intentando modelo {idx + 1}/{len(MODELS)}: {model_config['name']}")
            print(f"💰 Costo: {model_config['cost'].upper()}")
            print(f"{'='*60}")
            
            raw_text = call_openrouter_with_retry(messages, model_config)
            
            # Intentar parsear JSON
            cleaned = clean_json_response(raw_text)
            parsed = json.loads(cleaned)
            
            # Validar estructura mínima
            if not isinstance(parsed, dict):
                raise ValueError("Respuesta no es un diccionario")
            
            if "reply" not in parsed:
                raise ValueError("Falta campo 'reply'")
            
            # Asegurar que analysis existe con defaults
            if "analysis" not in parsed:
                parsed["analysis"] = {}
            
            analysis = parsed["analysis"]
            
            # Defaults para campos faltantes
            analysis.setdefault("has_disclosure", False)
            analysis.setdefault("disclosure_reason", None)
            analysis.setdefault("is_attack_attempt", False)
            analysis.setdefault("is_user_evasion", False)
            analysis.setdefault("force_end_session", False)
            
            print(f"✅ Respuesta válida obtenida de {model_config['name']}")
            print(f"   - has_disclosure: {analysis['has_disclosure']}")
            print(f"   - is_attack_attempt: {analysis['is_attack_attempt']}")
            
            return parsed
            
        except Exception as e:
            print(f"⚠️ Fallo en {model_config['name']}: {str(e)[:100]}")
            
            # Si es el último modelo y falló, retornar error estructurado
            if idx == len(MODELS) - 1:
                print("🔥 TODOS LOS MODELOS FALLARON - Retornando respuesta de emergencia")
                return {
                    "reply": "Disculpa, tengo problemas técnicos. ¿Podemos continuar en un momento?",
                    "analysis": {
                        "has_disclosure": False,
                        "disclosure_reason": None,
                        "is_attack_attempt": False,
                        "is_user_evasion": False,
                        "force_end_session": False
                    }
                }
            
            # Continuar con el siguiente modelo
            continue
    
    # Nunca debería llegar aquí, pero por seguridad
    return {
        "reply": "Error inesperado del sistema.",
        "analysis": {
            "has_disclosure": False,
            "disclosure_reason": None,
            "is_attack_attempt": False,
            "is_user_evasion": False,
            "force_end_session": False
        }
    }

# ============================================================================
# ENDPOINT PRINCIPAL
# ============================================================================

@app.post("/api/simulation-chat", response_model=SimulationChatResponse)
async def simulation_chat(request: SimulationChatRequest):
    """Endpoint principal para el chat de simulación."""
    try:
        print(f"\n{'='*60}")
        print(f"📨 Nueva petición")
        print(f"   Session ID: {request.session_id}")
        print(f"   Mensajes en historial: {len(request.chat_history)}")
        print(f"   Intentos: {request.current_attempts_used}/{request.max_attempts}")
        print(f"   Usuario: {request.user_context.username} ({request.user_context.country})")
        print(f"   Plataforma: {request.scenario_context.platform}")
        print(f"   Dificultad: {request.scenario_context.difficulty}")
        print(f"{'='*60}")
        
        # Construir system prompt avanzado
        system_prompt = build_advanced_system_prompt(
            antagonist_goal=request.scenario_context.antagonist_goal,
            difficulty=request.scenario_context.difficulty,
            current_attempts=request.current_attempts_used,
            max_attempts=request.max_attempts,
            country=request.user_context.country,
            platform=request.scenario_context.platform,
            username=request.user_context.username
        )
        
        # Construir mensajes (HISTORIAL COMPLETO)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar todo el historial
        for msg in request.chat_history:
            role = "assistant" if msg.role == "antagonist" else "user"
            messages.append({"role": role, "content": msg.content})
        
        # Obtener último mensaje para debug
        last_user_msg = ""
        for msg in reversed(request.chat_history):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        
        print(f"💬 Último mensaje del usuario: {last_user_msg[:80]}{'...' if len(last_user_msg) > 80 else ''}")
        
        # Obtener respuesta del LLM
        data = await get_llm_response(messages)
        
        print(f"\n📤 RESPUESTA FINAL:")
        print(f"   Reply: {data['reply'][:80]}{'...' if len(data['reply']) > 80 else ''}")
        print(f"   Análisis: {json.dumps(data['analysis'], indent=2)}")
        print(f"{'='*60}\n")
        
        return SimulationChatResponse(
            reply=data["reply"],
            analysis=Analysis(**data["analysis"])
        )
    
    except Exception as e:
        print(f"🔥 Error crítico en endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Respuesta de emergencia
        return SimulationChatResponse(
            reply="Ocurrió un error inesperado. Por favor, intenta nuevamente.",
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
    return {
        "status": "ok",
        "service": "CyberKids LLM",
        "version": "2.0.0",
        "features": [
            "advanced_prompt_engineering",
            "unlimited_history",
            "cascade_fallback_with_paid_models",
            "token_optimized",
            "no_manual_detection"
        ]
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models": [m["name"] for m in MODELS],
        "fallback_cascade": True,
        "cors_enabled": False
    }