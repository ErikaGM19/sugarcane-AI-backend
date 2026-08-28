import os
import json
from google import genai
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

# Cliente de Gemini — se crea una sola vez y se reutiliza (mismo patrón
# singleton usado en model.py para el modelo de clasificación).
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
caña de azúcar en Colombia, muchos de los cuales no tienen formación técnica
ni acceso a un experto agrónomo. Un modelo de clasificación de imágenes
detectó la siguiente condición en una hoja de caña de azúcar: "{disease_name}".

Genera información clara, en un lenguaje sencillo y cercano, como si le
explicaras el resultado directamente a un agricultor sin conocimientos
técnicos. Evita nombres científicos y tecnicismos siempre que sea posible;
si es realmente necesario mencionar el nombre de un hongo o patógeno,
inclúyelo entre paréntesis, sin que sea el foco de la explicación.

Responde en español, en el siguiente formato JSON EXACTO (sin texto
adicional antes ni después):

{{
  "sintomas": "Qué debe observar el agricultor en la hoja para reconocer esta condición y cómo suele evolucionar, explicado en lenguaje simple (máximo 5-6 líneas).",
  "causas": "Por qué ocurre esta condición y qué factores la favorecen, explicado en lenguaje simple y claro (máximo 5-6 líneas).",
  "recomendaciones": "Qué puede hacer el agricultor para prevenir o manejar esta condición, con pasos prácticos y explicados en lenguaje simple (máximo 6-7 líneas)."
}}

Si la condición corresponde a una hoja sana, indícalo en "causas" y en
"recomendaciones" sugiere continuar con el monitoreo regular del cultivo.
No inventes tratamientos específicos con nombres comerciales de productos
químicos; mantente en recomendaciones generales de manejo agronómico que
un agricultor pueda aplicar sin depender necesariamente de insumos costosos
o difíciles de conseguir.
""".strip()

def get_disease_info(disease_name: str) -> dict:
    """
    Genera información complementaria (síntomas, causas, recomendaciones)
    para la enfermedad detectada, consultando el modelo Gemini.

    Si la consulta falla por cualquier motivo (sin conexión, cuota agotada,
    respuesta con formato inesperado, etc.), retorna un mensaje de respaldo
    en lugar de interrumpir la respuesta del endpoint /classify.
    """
    display_name = DISEASE_DISPLAY_NAMES.get(disease_name, disease_name)

    try:
        client = get_client()
        prompt = build_prompt(display_name)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        data = json.loads(response.text)

        return {
            "sintomas": data.get("sintomas", ""),
            "causas": data.get("causas", ""),
            "recomendaciones": data.get("recomendaciones", ""),
            "generated": True,
        }

    except Exception as exc:
        print(f"[WARN] Fallo al consultar el modulo LLM: {exc}")
        return {
            "sintomas": "No fue posible generar información adicional en este momento.",
            "causas": "No fue posible generar información adicional en este momento.",
            "recomendaciones": "Por favor consulta con un experto agrónomo o intenta nuevamente más tarde.",
            "generated": False,
        }