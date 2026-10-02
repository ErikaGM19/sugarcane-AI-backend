import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Traduce el nombre interno de la clase (usado por el modelo de clasificación)
# al nombre correcto en español, tal como se documenta en la Tabla 2 del Capítulo 3.
DISEASE_DISPLAY_NAMES = {
    "Carbon": "Carbón",
    "Hoja Amarilla": "Hoja amarilla",
    "Mancha de Anillo": "Mancha de anillo",
    "Mancha Parda": "Mancha parda",
    "Mosaico": "Mosaico",
    "Pudricion Roja": "Muermo rojo",
    "Roya": "Roya",
    "Hoja Sana": "Hoja sana",
    "Hoja Seca": "Hoja seca",
    "Tizon Bacteriano": "Tizón bacteriano",
}

# Modelo principal y modelos alternativos por si alguno está saturado o sin cuota disponible.
MODELOS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
REINTENTOS_POR_MODELO = 1

# Cliente de Gemini — se crea una sola vez y se reutiliza (patrón singleton,
# igual que load_model() en model.py).
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no está configurada en el archivo .env")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(disease_name: str) -> str:
    return f"""
Eres un asistente agrícola que ayuda a pequeños y medianos agricultores de
caña de azúcar, muchos de los cuales no tienen formación técnica ni acceso a
un experto agrónomo.

Un modelo de clasificación de imágenes detectó la siguiente condición en la
hoja de caña de azúcar que se adjunta: "{disease_name}". Trabaja siempre a
partir de esta condición como un hecho ya establecido, sin cuestionarla ni
sugerir un diagnóstico distinto, incluso si algún detalle de la imagen no es
completamente claro.

Antes de generar la información, evalúa si la imagen muestra efectivamente
una hoja de caña de azúcar que podría presentar la condición indicada. Si la
imagen claramente no corresponde a una hoja de este tipo —por ejemplo, si
muestra una persona, un animal, un vehículo, un objeto, u otro tipo de
planta evidentemente distinta—, indica esto mediante el campo "hoja_valida"
con el valor false. En ese caso, escribe en el campo "mensaje" un único
texto breve y sencillo (máximo 3-4 líneas) que explique que la imagen no
parece corresponder a una hoja de caña de azúcar —indicando, si es evidente,
qué muestra realmente la imagen— y que sugiera cargar una fotografía
adecuada, sin describir ninguna enfermedad; deja "sintomas", "causas" y
"recomendaciones" como cadenas vacías. Si la imagen sí corresponde
razonablemente a una hoja de caña de azúcar, utiliza "hoja_valida": true,
deja "mensaje" como cadena vacía y continúa con la respuesta normal.

Observa la imagen proporcionada y utiliza lo que se ve en ella —como la
extensión del daño, la intensidad del color de las lesiones, o si el
deterioro parece inicial o avanzado— para que tu respuesta sea específica a
este caso particular, y no una descripción genérica de la enfermedad. Ten en
cuenta que la imagen muestra únicamente una hoja: si existe un riesgo
conocido de que esta condición también afecte otras partes de la planta
(como el tallo), puedes mencionarlo como algo a verificar, pero nunca como
algo que ya se observa en la fotografía.

Genera información clara, en un lenguaje sencillo y cercano, como si le
explicaras el resultado directamente a un agricultor sin conocimientos
técnicos. Evita nombres científicos y tecnicismos siempre que sea posible;
si es realmente necesario mencionar el nombre de un hongo, virus o
patógeno, inclúyelo entre paréntesis, sin que sea el foco de la explicación.

Para las recomendaciones, ajusta su alcance según la gravedad observada en
la imagen: si el daño es leve o está localizado en una zona pequeña,
prioriza acciones simples y puntuales, como retirar el tejido afectado y
continuar con el monitoreo del cultivo; si el daño es extenso o compromete
gran parte de la hoja, incluye además medidas de mayor alcance orientadas a
frenar la propagación hacia otras plantas, como evitar usar ese lote como
fuente de semilla para siembras futuras, dar prioridad a su manejo o
cosecha, o reforzar el control de los factores que favorecen la enfermedad.
Usa siempre medidas razonables y conocidas para el tipo de condición
identificada, sin inventar prácticas que no tengan sustento agronómico.

Responde en español, en el siguiente formato JSON EXACTO (sin texto
adicional antes ni después):

{{
  "hoja_valida": true o false, según corresponda,
  "mensaje": "Solo cuando hoja_valida sea false: un texto breve y sencillo que explique qué se ve en la imagen y por qué no corresponde a una hoja de caña de azúcar y sugiera cargar otra fotografía. Si hoja_valida es true, déjalo como cadena vacía.",
  "sintomas": "Describe lo que se observa específicamente en esta imagen (extensión, color, apariencia de las lesiones) y cómo suele evolucionar este tipo de condición, en lenguaje simple (máximo 5-6 líneas).",
  "causas": "Por qué ocurre esta condición y qué factores la favorecen, explicado en lenguaje simple y claro (máximo 5-6 líneas).",
  "recomendaciones": "Qué puede hacer el agricultor para prevenir o manejar esta condición, con pasos prácticos acordes a la gravedad observada, explicados en lenguaje simple (máximo 6-7 líneas)."
}}

Si la condición corresponde a una hoja sana, indícalo en "causas" y en
"recomendaciones" sugiere continuar con el monitoreo regular del cultivo.
No inventes tratamientos específicos con nombres comerciales de productos
químicos; mantente en recomendaciones generales de manejo agronómico que
un agricultor pueda aplicar sin depender necesariamente de insumos costosos
o difíciles de conseguir.
""".strip()


def _es_error_temporal(exc: Exception) -> bool:
    """Indica si el error es temporal (servicio saturado) y vale la pena reintentar."""
    texto = str(exc)
    return "503" in texto or "UNAVAILABLE" in texto


def get_disease_info(disease_name: str, image_bytes: bytes, mime_type: str) -> dict:
    """
    Genera información complementaria (síntomas, causas, recomendaciones)
    para la enfermedad detectada, enviando a Gemini el nombre de la
    enfermedad y la imagen original. El propio modelo determina, a partir
    de la imagen, si esta corresponde efectivamente a una hoja de caña de
    azúcar (campo "recognized").

    Si el modelo principal está saturado (error 503), reintenta y luego
    prueba con un modelo alternativo. Si todo falla, retorna un mensaje de
    respaldo, asumiendo que la imagen sí es válida (recognized=True), ya
    que en ese caso no fue posible verificarlo con el LLM.
    """
    display_name = DISEASE_DISPLAY_NAMES.get(disease_name, disease_name)

    respaldo = {
        "sintomas": "No fue posible generar información adicional en este momento.",
        "causas": "No fue posible generar información adicional en este momento.",
        "recomendaciones": "Por favor consulta con un experto agrónomo o intenta nuevamente más tarde.",
        "generated": False,
        "recognized": True,
        "mensaje": "",
   }

    try:
        client = get_client()
        prompt = build_prompt(display_name)
        if image_bytes is not None:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            contents = [prompt, image_part]
        else:
            contents = prompt
    except Exception as exc:
        print(f"[WARN] Fallo al preparar la consulta al modulo LLM: {exc}")
        return respaldo

    for modelo in MODELOS:
        for intento in range(1, REINTENTOS_POR_MODELO + 1):
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=contents,
                    config={"response_mime_type": "application/json"},
                )
                data = json.loads(response.text)
                print(f"[INFO] Respuesta generada con {modelo}")

                hoja_valida = data.get("hoja_valida", True)
                if isinstance(hoja_valida, str):
                    hoja_valida = hoja_valida.strip().lower() in ("true", "1", "si", "sí")
                else:
                    hoja_valida = bool(hoja_valida)

                mensaje = str(data.get("mensaje", "") or "").strip()
                if not hoja_valida and not mensaje:
                    mensaje = (
                        "La imagen cargada no parece corresponder a una hoja de caña de azúcar. "
                        "Por favor, carga una fotografía clara de una hoja de caña de azúcar."
                    )

                return {
                    "sintomas": data.get("sintomas", ""),
                    "causas": data.get("causas", ""),
                    "recomendaciones": data.get("recomendaciones", ""),
                    "mensaje": mensaje,
                    "generated": True,
                    "recognized": hoja_valida,
                }
            
            except Exception as exc:
                print(f"[WARN] {modelo}, intento {intento}: {exc}")
                if _es_error_temporal(exc) and intento < REINTENTOS_POR_MODELO:
                    time.sleep(2 * intento)  # espera 2 s, luego 4 s
                    continue
                # Si falló este intento (por 429 cuota, 503 tras reintentos, etc.),
                # pasamos al siguiente modelo de la lista en lugar de abortar inmediatamente.
                break

    return respaldo