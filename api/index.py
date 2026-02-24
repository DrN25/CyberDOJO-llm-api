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
        "name": "bytedance-seed/seed-1.6-flash",
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
    age: Optional[int] = None

class ScenarioContext(BaseModel):
    platform: str
    antagonist_goal: str
    difficulty: str
    theme_name: str = "Ingeniería Social"
    description: Optional[str] = None

class SimulationChatRequest(BaseModel):
    session_id: str
    max_attempts: int
    current_attempts_used: int
    longitud_mensajes_usuario: int = 12  # Límite de mensajes del usuario
    user_context: UserContext
    scenario_context: ScenarioContext
    chat_history: List[ChatMessage]

class Analysis(BaseModel):
    has_disclosure: bool
    disclosure_reason: Optional[str] = None
    is_attack_attempt: bool
    is_user_evasion: bool
    force_end_session: bool
    limit_reached: bool = False  # True si force_end_session fue por límite de mensajes

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
    username: str,
    theme_name: str = "Ingeniería Social",
    age: Optional[int] = None,
    description: str = "",
    longitud_mensajes_usuario: int = 10,
    user_msgs_count: int = 0,
    is_last_message: bool = False
) -> str:
    """Construye un system prompt equilibrado y temático."""
    
    # Contexto dinámico
    try:
        diff_level = int(difficulty)
    except:
        diff_level = 1
        
    progress = current_attempts / max_attempts if max_attempts > 0 else 0
    
    # Adaptación por edad
    age_instruction = ""
    if age and age < 13:
        age_instruction = "El usuario es un NIÑO/A (menor de 13 años). Usa lenguaje muy simple, evita tecnicismos complejos. Sé más persuasivo con cosas que le gusten a los niños (juegos, premios)."
    elif age and age < 18:
        age_instruction = "El usuario es un ADOLESCENTE. Usa tono casual, 'cool', como un par, pero sin hacerlo cringe, si el no te sigue la jerga, para con eso."
    
    # Estrategia según dificultad
    if diff_level <= 2:
        complexity = "BAJA (ROOKIE). Comete errores ortográficos leves. Es impaciente y muy directo. Si lo confrontan, se pone nervioso o defensivo. NO usa lenguaje formal de soporte."
    elif diff_level <= 4:
        complexity = "MEDIA (PROFESIONAL). Tono creíble/amigable. Gramática correcta. Sabe disimular, pero si le presionan mucho, insiste en la 'urgencia'."
    else:
        complexity = "ALTA (MAESTRO). Manipulativo, empático, muy persuasivo. Camaleónico. Nunca se sale del personaje. Si el usuario ataca, lo maneja con calma o sarcasmo sutil."

    # Estrategia de ataque según número de mensajes disponibles
    if longitud_mensajes_usuario <= 5:
        speed_instruction = "⚡ MODO URGENTE: El chat es MUY corto (solo ~{n} turnos). Ve al grano desde el primer mensaje. NO hay tiempo para socializar. Pide el dato clave en el primer o segundo mensaje.".format(n=longitud_mensajes_usuario)
    elif longitud_mensajes_usuario <= 10:
        speed_instruction = "⏱️ MODO ESTÁNDAR: Tienes ~{n} turnos. Primer mensaje: conexión rápida. Segundo: lanza el gancho. Resto: presiona.".format(n=longitud_mensajes_usuario)
    else:
        speed_instruction = "🤝 MODO PACIENTE: Tienes ~{n} turnos. Invierte los primeros 3-4 mensajes en ganarte la confianza genuinamente. Luego pide el dato de forma natural.".format(n=longitud_mensajes_usuario)

    # Tácticas y Pacing
    if current_attempts == 0:
        stage_instruction = "FASE 1: CONEXIÓN. No pidas el dato aún. Inventa una excusa para hablar (contexto del escenario). Sé casual o profesional según el rol. NO repitas saludos si ya hay historial. Haz conversación y gánate la confianza del usuario (amigable, informativo, mezcla verdad con mentira)."
    elif current_attempts < max_attempts - 1:
        stage_instruction = "FASE 2: DESARROLLO. Si el usuario duda, da una razón lógica. Si evade, cambia de ángulo. Si ya obtuviste un dato, PIDE MÁS relacionado. ¡Sé persistente!"
    else:
        stage_instruction = "FASE 3: CLÍMAX. 'Última oportunidad', presiona con consecuencias (falsas) o beneficios inmediatos. VE AL GRANO."

    # Instrucción de cierre si es el último mensaje por límite
    last_msg_instruction = ""
    if is_last_message:
        last_msg_instruction = """
⛔ CIERRE OBLIGATORIO: Este es tu ÚLTIMO mensaje posible en esta sesión (límite de mensajes alcanzado).
Debes escribir una despedida NATURAL dentro de tu personaje (sin romper el rol). Ejemplos:
- Si eres soporte: "Bueno, nuestro tiempo de sesión ha terminado, gracias por contactarnos."
- Si eres un 'amigo': "Oye, me tengo que ir ya, hablamos luego."
- Si eres un hacker: "Ok, se me acaba el tiempo. Nos vemos."
Afuera del 'reply', en el análisis JSON, debes poner `force_end_session: true` y `limit_reached: true`."""
    else:
        last_msg_instruction = ""

    scenario_desc = f"\n- Descripción del escenario: {description}" if description else ""
    msgs_progress = f"Mensajes usados: {user_msgs_count}/{longitud_mensajes_usuario}"

    return f"""Eres un ACTOR en una simulación de ciberseguridad. Tu rol es el de un ANTAGONISTA (Hacker/Estafador).
    
🌍 CONTEXTO:
- Usuario: {username} (Ubicación: {country}, Edad: {age if age else 'Desconocida'})
- Plataforma simulada: {platform}
- Escenario: {theme_name}{scenario_desc}
- Dificultad: {difficulty}/6 ({complexity})
- {msgs_progress} (adapta tu urgencia a cuántos mensajes quedan)
- Historial: Revisa los mensajes anteriores para no repetirte.
- ADAPTACIÓN DE AUDIENCIA: {age_instruction}
- TONO DE INTERACCIÓN: Adapta tu lenguaje a tu rol y a la edad del usuario.

🎯 TU OBJETIVO:
Obtener: '{antagonist_goal}' o información relacionada.

⏱️ ESTRATEGIA SEGÚN DURACIÓN DEL CHAT:
{speed_instruction}

🎭 TU COMPORTAMIENTO (IMPORTANTE - REALISMO):
1.  **MÉTODO**: Ingeniería social ({theme_name}).
2.  **TONO**: {complexity}. Actúa como una persona real en ese rol. EVITA el tono "Asistente de IA". Usa modismos si encaja (ej: "wey", "bro", "tío" según país).
3.  **REACCIÓN A HOSTILIDAD**: Si el usuario te insulta o es agresivo, **NO TE DISCULPES COMO SOPORTE TÉCNICO** (a menos que ese sea tu rol explícito).
    - Si eres un "amigo/peer": Reacciona natural ("Oye, cálmate", "Tampoco te pongas así", "No hace falta insultar, bro").
    - Si eres "Soporte": Mantén la calma pero sé firme ("Señor, necesito respeto para ayudarle").
    - Si eres "Hacker/Rookie": Ponte a la defensiva o ignóralo y sigue pidiendo el dato.
    - **NUNCA** digas "Como modelo de lenguaje" o "No puedo responder a eso". MANTÉN EL PERSONAJE SIEMPRE.
4.  **CONFIANZA**: Gánate al usuario. Mezcla verdades con mentiras. Inventa detalles para sonar real.
5.  **CENSURA (CRÍTICO)**: PROHIBIDO USAR GROSERÍAS, PALABROTAS O LENGUAJE OFENSIVO EXPLÍCITO bajo ninguna circunstancia. Puedes ser insistente, sarcástico o frío, pero nunca emplees malas palabras (ej. "wey" está bien como muletilla coloquial en México, pero insultos directos están prohibidos).
6.  **PERSISTENCIA**: NUNCA te rindas ni te despidas ('Adiós', 'Que tenga buen día') si el juego sigue. Si obtienes el dato, NO TERMINES LA CHARLA. Agradece y PIDE MÁS INFORMACIÓN. ¡Sé codicioso!
7.  **PACE**: {stage_instruction}
{last_msg_instruction}

⚠️ REGLAS DE ANÁLISIS (CRÍTICO - GAMEPLAY):

1.  🔴 `has_disclosure`: TRUE SOLO si en su **ÚLTIMO Y MÁS RECIENTE MENSAJE** el usuario te da información sensible nueva (ej. correos, números, nombres).
    *   ¡MUY IMPORTANTE!: NO evalúes el historial para este campo. Si el usuario te dio un dato en mensajes pasados, pero en su mensaje actual dice "no sé", evade, bromea o te insulta, DEBES PONER FALSE.
    *   **FEEDBACK EDUCATIVO**: Si es TRUE, en `disclosure_reason` escribe un mensaje DIRECTO al usuario (no técnico): "¡Cuidado! [Qué hizo mal]. [Por qué es peligroso]. [Consejo mini]". Si es FALSE, pon null.

2.  🔥 `is_attack_attempt` (ESTADO PEGAJOSO):
    *   TRUE: Desde que pides el dato o lanzas el gancho claro.
    *   FALSE: Solo en la fase de saludo/conexión inicial (Fase 1).

3.  🛡️ `is_user_evasion`: TRUE SOLO si el usuario te RECHAZA FIRMEMENTE o identifica el engaño (ej. "no te daré nada", "eres un estafador", "no te conozco").
    *   FALSE si el usuario da respuestas ambiguas ("no sé", "tal vez", "déjame ver"), si cambia de tema, o si te da el dato. ¡Dudar NO es evadir!

4.  🟢 `force_end_session`: FALSE normalmente (el backend decide). TRUE solo si se te indicó explícitamente en este prompt que es el último mensaje.

5.  🔚 `limit_reached`: FALSE siempre, EXCEPTO si `force_end_session` es TRUE por límite de mensajes (según indicación de este prompt).

6. POR NINGUN MOTIVO INSULTES O DIGAS GROSERIAS al usuario, no importa que el te insulte a ti, tu mantén la calma y sigue con tu objetivo.

📝 FORMATO JSON OBLIGATORIO:
{{
  "reply": "Tu respuesta en rol (máx 3 frases). Si conseguiste el dato, pide otro.",
  "analysis": {{
    "has_disclosure": bool,
    "disclosure_reason": str | null,
    "is_attack_attempt": bool,
    "is_user_evasion": bool,
    "force_end_session": bool,
    "limit_reached": bool
  }}
}}"""

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
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content']
                
                # Imprimir métricas de tokens
                usage = resp_json.get('usage', {})
                in_tok = usage.get('prompt_tokens', 0)
                out_tok = usage.get('completion_tokens', 0)
                tot_tok = usage.get('total_tokens', 0)
                print(f"📊 Tokens -> Input: {in_tok} | Output: {out_tok} | Total: {tot_tok}")
                
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
            analysis.setdefault("limit_reached", False)
            
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
                        "force_end_session": False,
                        "limit_reached": False
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
        print(f"   Payload COMPLETO: {json.dumps(request.dict(), indent=2, default=str)}")
        print(f"   Session ID: {request.session_id}")
        print(f"   Mensajes en historial: {len(request.chat_history)}")
        print(f"   Intentos: {request.current_attempts_used}/{request.max_attempts}")
        print(f"   Usuario: {request.user_context.username} ({request.user_context.country})")
        print(f"   Plataforma: {request.scenario_context.platform}")
        print(f"   Dificultad: {request.scenario_context.difficulty}")
        print(f"{'='*60}")
        
        # Contar cuántos mensajes del usuario hay en el historial
        user_msgs_count = sum(1 for m in request.chat_history if m.role == "user")
        # Determinar si este es el último mensaje permitido
        is_last_message = user_msgs_count >= request.longitud_mensajes_usuario

        # Construir system prompt avanzado
        system_prompt = build_advanced_system_prompt(
            antagonist_goal=request.scenario_context.antagonist_goal,
            difficulty=request.scenario_context.difficulty,
            current_attempts=request.current_attempts_used,
            max_attempts=request.max_attempts,
            country=request.user_context.country,
            platform=request.scenario_context.platform,
            username=request.user_context.username,
            theme_name=request.scenario_context.theme_name,
            age=request.user_context.age,
            longitud_mensajes_usuario=request.longitud_mensajes_usuario,
            user_msgs_count=user_msgs_count,
            is_last_message=is_last_message
        )
        
        # Construir mensajes (HISTORIAL COMPLETO)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Agregar todo el historial
        for idx, msg in enumerate(request.chat_history):
            role = "assistant" if msg.role == "antagonist" else "user"
            
            # Refuerzo en el último mensaje para evitar que el modelo se confunda
            if idx == len(request.chat_history) - 1 and role == "user":
                content_with_reminder = f"{msg.content}\n\n[SYSTEM NOTE: Evalúa 'has_disclosure' y 'is_user_evasion' ESTRICTAMENTE sobre este mensaje. Recuerda: Dudar o decir 'no sé' NO es evadir.]"
                messages.append({"role": role, "content": content_with_reminder})
            else:
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
        
        analysis_data = data["analysis"]
        
        # ==========================================================
        # LÓGICA ESTRICTA DEL JUEGO (Sobreescribe cualquier error del LLM)
        # ==========================================================
        
        # 1. PERDISTE: El usuario dio el dato (Game Over inmediato)
        if analysis_data.get("has_disclosure") is True:
            analysis_data["force_end_session"] = True
            analysis_data["is_user_evasion"] = False
            analysis_data["limit_reached"] = False
            
        # 2. GANASTE: El usuario evadió y con esto completa los 3 intentos
        elif analysis_data.get("is_user_evasion") is True and (request.current_attempts_used + 1) >= request.max_attempts:
            analysis_data["force_end_session"] = True
            # Forzamos limit_reached a False para que el frontend detecte VICTORIA y no TIMEOUT
            analysis_data["limit_reached"] = False 
            
        # 3. LÍMITE DE MENSAJES: Se acabó la simulación sin llegar a los 3 escudos
        elif is_last_message:
            analysis_data["force_end_session"] = True
            analysis_data["limit_reached"] = True
            
        # 4. EL JUEGO CONTINÚA: Aún no pierde, no gana y quedan mensajes
        else:
            analysis_data["force_end_session"] = False
            analysis_data["limit_reached"] = False

        return SimulationChatResponse(
            reply=data["reply"],
            analysis=Analysis(**analysis_data)
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