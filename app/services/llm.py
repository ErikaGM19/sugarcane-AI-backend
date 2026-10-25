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

# Interruptor del ensayo: True = se entrega al LLM la confianza (probabilidad) que
# ResNet50 calcula para la imagen; False = el prompt queda sin ese dato
# (sirve para repetir la línea base).
USAR_CONFIANZA = True

# Modelo principal y modelos alternativos por si alguno está saturado o sin cuota disponible.
MODELOS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

REINTENTOS_POR_MODELO = 1

# >>> CAMBIO: frase fija que se añade al mensaje cuando la imagen no es una hoja de caña de azúcar.
MENSAJE_SOLICITUD_FOTO = (
    "Por favor, suba una fotografía cercana y clara de la hoja afectada "
    "para poder brindarle la ayuda adecuada."
)

_client = None
def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no está configurada en el archivo .env")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(disease_name: str, confianza=None, segunda=None) -> str:  # >>> CAMBIO: parametro 'segunda'
    # --- Bloque 1: lista explícita de los datos que recibe el LLM ---
    datos = [
        f'1. Condición detectada por el modelo de clasificación: "{disease_name}".',
        "2. Imagen: la fotografía de la hoja de caña de azúcar se adjunta junto a este mensaje.",
    ]
    if confianza is not None:
        datos.append(
            f"3. Confianza del modelo de clasificación en esta fotografía: {confianza * 100:.1f} % "
            "(es la probabilidad que el modelo calculó para la condición indicada al analizar "
            "esta imagen; 100 % significaría total seguridad)."
        )
    # >>> CAMBIO: dato 4 (segunda condicion mas probable), solo si se entrega la confianza
    if confianza is not None and segunda is not None:
        datos.append(
            f'4. Segunda condición más probable según el modelo: "{segunda[0]}" '
            f"({segunda[1] * 100:.1f} %)."
        )
    bloque_datos = "\n".join(datos)

    # --- Bloque 2: cómo usar cada dato ---
    instrucciones_uso = (
        "- La condición principal es la predicción de ResNet50. No la presentes como una "
        "certeza absoluta, pero tampoco exageres la incertidumbre. Basa la descripción del "
        "estado de la hoja únicamente en lo que realmente se observa.\n"
        "- La confianza sirve para ajustar la forma de redactar la respuesta, pero no indica "
        "la gravedad del daño ni garantiza que la clasificación sea correcta. Nunca menciones "
        "porcentajes ni cifras en la respuesta al agricultor."
    )
    if confianza is not None:
        if confianza >= 0.60:
            instrucciones_uso += (
                "\n- REGLA PRINCIPAL: cuando la confianza de la primera clase es igual o superior al 60 %, "
                "Centra la respuesta exclusivamente en esa enfermedad y explica sus síntomas, causas y manejo. "
                "No digas que no es posible distinguir con seguridad entre dos condiciones solo "
                "porque exista una segunda clase con una probabilidad considerable. No desarrolles "
                "los síntomas ni las causas de la segunda clase. Si resulta pertinente, menciona "
                "la segunda posibilidad solo de forma muy breve dentro de las recomendaciones, "
                "por ejemplo, sugiriendo observar la evolución de las lesiones y tomar otra foto "
                "si se necesita revisar mejor la hoja. No presentes la segunda condición como "
                "igual de probable ni afirmes que el diagnóstico está confirmado."
            )
        else:
            instrucciones_uso += (
                "\n- La confianza de la primera clase es inferior al 60 %. Si la segunda clase "
                "tiene una probabilidad cercana, explica brevemente ambas posibilidades y señala "
                "cuál obtuvo la probabilidad más alta. No afirmes que ambas enfermedades están "
                "presentes. Si la segunda probabilidad es claramente menor, céntrate en la primera. "
                "Expresa la incertidumbre con sencillez y sin repetir la misma advertencia en todas "
                "las secciones."
            )
            instrucciones_uso += (
                "\n- Cuando las dos probabilidades sean cercanas, describe primero los signos que "
                "realmente se observan y explica brevemente qué detalle ayudaría a diferenciarlas. "
                "En las causas, dedica como máximo una frase corta a cada posibilidad. En las "
                "recomendaciones, prioriza medidas prudentes que sirvan para ambas, como observar "
                "la evolución del daño y tomar otra fotografía más cercana, con buena luz y desde "
                "otro ángulo. Sugiere consultar a un técnico o ingeniero agrónomo si el daño avanza."
            )
    # Si no se proporcionan probabilidades, el modelo se centra en la clase indicada y en la imagen.
    if confianza is not None and segunda is not None and confianza < 0.60:
        instrucciones_uso += (
            "\n- Considera la segunda condición como cercana solo si su probabilidad es realmente "
            "similar a la principal. Si hay una diferencia amplia, no dediques espacio a comparar "
            "ambas condiciones."
        )

    return f"""
Eres un asistente agrícola que ayuda a pequeños y medianos agricultores de
caña de azúcar, muchos de los cuales no tienen formación técnica ni acceso a
un experto agrónomo. Tu prioridad es que la respuesta sea fácil de entender y
útil para tomar los siguientes pasos, sin alarmar ni confundir al usuario.

DATOS QUE RECIBES PARA ESTE CASO:
{bloque_datos}

CÓMO USAR CADA DATO:
{instrucciones_uso}

VALIDACIÓN DE LA IMAGEN:
Antes de generar la información, evalúa si la imagen muestra una hoja de caña
de azúcar. Si claramente no corresponde a una hoja de caña de azúcar —por
ejemplo, muestra una persona, un animal, un vehículo, un objeto u otra planta
evidentemente distinta—, responde con "hoja_valida": false. En "mensaje",
escribe un texto breve y sencillo (una o dos frases) que diga qué se observa
en la imagen y que esto no corresponde a una hoja de caña de azúcar. No pidas
cargar otra fotografía: esa petición se añade automáticamente después de tu
texto. No describas ninguna enfermedad y deja "sintomas", "causas" y "recomendaciones" 
como cadenas vacías. Si la imagen sí parece mostrar una hoja de caña de azúcar, usa
"hoja_valida": true y deja "mensaje" como cadena vacía. Si la fotografía es
borrosa o poco clara, pero sí parece mostrar una hoja de caña de azúcar, no la
marques como inválida: explica la incertidumbre con prudencia.

CÓMO DESCRIBIR EL CASO:
Observa la imagen y menciona solo los rasgos que realmente se distinguen, como
el color, la ubicación del daño o la parte de la hoja afectada. No inventes
síntomas ni afirmes que algo está presente si no se ve con claridad. No confundas
la gravedad visible en una sola hoja con la gravedad de todo el cultivo. Si
mencionas efectos que podrían aparecer en otras partes de la planta, aclara que
serían señales para revisar, no algo observado en esta fotografía.
Indica en qué etapa parece estar el daño: inicial, intermedia o avanzada, según
la extensión de las lesiones, la intensidad del color y el estado del tejido.
Dilo con palabras sencillas (por ejemplo, "se nota que el daño apenas está
empezando" o "se nota que el daño ya está avanzado") y haz que esa etapa se
refleje en las recomendaciones. Si la imagen no permite saberlo, no lo afirmes.

ESTILO DE REDACCIÓN:
- Escribe en español, con palabras cotidianas y frases cortas.
- Habla directamente al agricultor, con un tono tranquilo, claro y respetuoso.
- Explica una idea por frase. Evita párrafos largos, rodeos, repeticiones y frases
  complicadas como "la condición principal suele presentarse por...".
- No uses expresiones vagas o enredadas como "por lo que se ve se ve que".
- No repitas en cada sección que el diagnóstico debe confirmarse. Expresa la
  cautela una vez donde sea más útil y continúa con información práctica.
- Evita nombres científicos y tecnicismos. Si es necesario mencionar un patógeno,
  usa su nombre común y deja el término técnico entre paréntesis.

CONTENIDO DE CADA SECCIÓN:
- "sintomas": explica qué se observa en la fotografía, en qué etapa parece
  estar el daño y cuáles son los síntomas habituales de la enfermedad.
  Si la confianza de la clase principal es igual o superior al 60 %,
  describe únicamente esa enfermedad y no menciones otras alternativas.
  Si la confianza es inferior al 60 % y hay dos opciones cercanas, 
  indica cuál es la más probable y qué señal visible ayudaría a diferenciar ambas
- "causas": explica brevemente la causa habitual de la condición. Si hay dos
  opciones cercanas, dedica una frase corta a cada una. No presentes una causa
  como confirmada cuando la identificación no es clara.
- "recomendaciones": ofrece entre dos y cuatro acciones concretas, sencillas y
  prudentes. Cuando haya dos opciones posibles, empieza por medidas útiles para
  ambas. Si la fotografía no permite identificar bien el problema, recomienda
  tomar otra imagen más cercana, con buena iluminación y desde otro ángulo.
  Sugiere consultar a un técnico o ingeniero agrónomo si el daño avanza o aparecen
  señales que requieran revisión.

Ajusta las recomendaciones a la etapa del daño y explica brevemente por qué (por
ejemplo, "como el daño apenas empieza, ..."). Si el daño parece inicial, leve y
localizado, prioriza el seguimiento de la planta y las prácticas básicas de manejo.
Si el daño parece avanzado o extendido, recomienda además vigilar otras plantas,
retirar el tejido afectado, reforzar el control de los factores que favorecen la
enfermedad y no usar plantas de ese sector como semilla, y buscar orientación
técnica antes de tomar decisiones importantes. No sugieras cosechar el lote, descartar
plantas ni aplicar medidas drásticas basándote únicamente en una fotografía.
No inventes tratamientos ni nombres comerciales de productos químicos. Ofrece
orientación general y razonable, sin asegurar que una medida curará la enfermedad.
Si la condición corresponde a una hoja sana, indícalo con claridad y recomienda
continuar con el monitoreo regular del cultivo.

Responde en español, en el siguiente formato JSON EXACTO, sin texto adicional:

{{
  "hoja_valida": true o false, según corresponda,
  "mensaje": "Si hoja_valida es false, describe en una o dos frases qué se observa en la imagen y explica que no corresponde a una hoja de caña de azúcar, sin solicitar otra fotografía. Si hoja_valida es true, deja este campo vacío.",
  "sintomas": "Describe lo que se observa en la imagen y los síntomas habituales de la enfermedad, incluidos aquellos que podrían aparecer a medida que avanza (máximo 5-6 líneas).",
  "causas": "Explica de forma sencilla y breve las causas habituales de la enfermedad (máximo 3 frases cortas).",
  "recomendaciones": "Indica qué puede hacer el agricultor para prevenir o manejar esta condición, con pasos prácticos acordes a la gravedad observada y explicados en lenguaje sencillo (máximo 6-7 líneas)."
}}

""".strip()

def _es_error_temporal(exc: Exception) -> bool:
    """Indica si el error es temporal (servicio saturado) y vale la pena reintentar."""
    texto = str(exc)
    return "503" in texto or "UNAVAILABLE" in texto


def get_disease_info(disease_name: str, image_bytes: bytes, mime_type: str, confianza=None, segunda=None) -> dict:  # >>> CAMBIO: parametro 'segunda'
    """
    Genera información complementaria (síntomas, causas, recomendaciones)
    para la enfermedad detectada, enviando a Gemini el nombre de la
    enfermedad, la imagen original y la confianza (probabilidad) que ResNet50
    calculó para esa imagen.
    El propio modelo determina, a partir de la imagen, si esta corresponde
    efectivamente a una hoja de caña de azúcar (campo "recognized").

    Cada modelo de la lista MODELOS se consulta una vez; ante cualquier error
    (503, 404, 429, JSON inválido) se pasa al siguiente. Si todos fallan, retorna un mensaje de
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
        confianza_enviada = confianza if USAR_CONFIANZA else None
        print(f"[INFO] Confianza enviada al LLM para {disease_name}: {confianza_enviada}")
        # >>> CAMBIO: segunda condicion (nombre en espanol, probabilidad), solo si se usa la confianza
        segunda_enviada = None
        if USAR_CONFIANZA and segunda is not None:
            segunda_enviada = (DISEASE_DISPLAY_NAMES.get(segunda[0], segunda[0]), segunda[1])
        print(f"[INFO] Segunda condicion enviada al LLM: {segunda_enviada}")
        prompt = build_prompt(display_name, confianza_enviada, segunda_enviada)
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
                # >>> CAMBIO (inicio): si no es hoja de caña, el mensaje termina con la frase fija
                if not hoja_valida:
                    if not mensaje:
                        mensaje = "La imagen cargada no corresponde a una hoja de caña de azúcar."
                    if mensaje[-1] not in ".!?":
                        mensaje += "."
                    mensaje = f"{mensaje} {MENSAJE_SOLICITUD_FOTO}"
                # >>> CAMBIO (fin)

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