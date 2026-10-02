import os
import io
from PIL import Image as PILImage
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.model import predict
from app.services.llm import get_disease_info
from app.dependencies import get_current_user

router = APIRouter()

# Tipos MIME que se aceptan sin restricción adicional.
# PIL soporta en su mayoría todos estos; los que no son JPEG/PNG/WEBP
# se convierten internamente antes de pasar al modelo.
ALLOWED_TYPES = {
    # JPEG y variantes (Windows, móviles, navegadores)
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/jfif",
    # Otros formatos comunes de cámara y escritorio
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/x-bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "image/x-tiff",
    "image/heic",
    "image/heif",
    "image/avif",
    # Aplicaciones que envían tipo genérico
    "application/octet-stream",
}
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".jfif",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tiff", ".tif",
    ".heic", ".heif",
    ".avif",
}
MAX_SIZE_MB = 30

# Umbral mínimo de confianza para aceptar una predicción como válida.
# Por debajo de esto, se asume que la imagen probablemente no es una
# hoja de caña de azúcar (u otra condición no vista en entrenamiento).
CONFIDENCE_THRESHOLD = 0.90

# Información de respaldo cuando la confianza es demasiado baja.
# No se consulta a Gemini en este caso para ahorrar cuota y latencia.
UNRECOGNIZED_INFO = {
    "sintomas": "No fue posible asociar la imagen con ninguna de las enfermedades conocidas con suficiente certeza.",
    "causas": "Esto puede deberse a que la imagen no corresponde a una hoja de caña de azúcar, o a que la fotografía tiene mala iluminación, enfoque o encuadre.",
    "recomendaciones": "Verifica que la imagen sea de una hoja de caña de azúcar, tomada con buena luz y ocupando la mayor parte del encuadre. Si el problema persiste, consulta con un experto agrónomo.",
    "generated": False,
    "recognized": False,
}

def _normalize_to_jpeg(raw_bytes: bytes) -> tuple[bytes, str]:
    """
    Convierte cualquier imagen soportada por Pillow a JPEG RGB.
    Retorna (jpeg_bytes, 'image/jpeg').
    Lanza ValueError si los bytes no corresponden a una imagen válida.
    """
    try:
        img = PILImage.open(io.BytesIO(raw_bytes))
        # Normalizar a RGB (elimina canal alfa, maneja paletas, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "image/jpeg"
    except Exception as exc:
        raise ValueError(f"El archivo no pudo abrirse como imagen: {exc}") from exc


@router.post("/classify")
async def classify_image(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    content_type = (file.content_type or "").lower()

    # Validar tipo de archivo y extensión
    if content_type not in ALLOWED_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no soportado ({content_type or ext or 'desconocido'}). "
                f"Se aceptan: JPG, PNG, WEBP, GIF, BMP, TIFF y HEIC."
            )
        )

    image_bytes = await file.read()

    # Validar que no esté vacío
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="El archivo seleccionado está vacío (0 bytes). Por favor selecciona una imagen válida."
        )

    # Validar tamaño
    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        size_mb = round(len(image_bytes) / (1024 * 1024), 1)
        raise HTTPException(
            status_code=400,
            detail=f"La imagen pesa {size_mb}MB y supera el límite de {MAX_SIZE_MB}MB permitidos."
        )

    # Normalizar a JPEG antes de la inferencia.
    # Esto garantiza compatibilidad con PIL/TensorFlow para formatos como
    # GIF (primer frame), BMP, TIFF o HEIC, y elimina canales alfa problemáticos.
    try:
        image_bytes, content_type = _normalize_to_jpeg(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar el archivo como imagen. Verifica que el archivo no esté dañado. ({exc})"
        )

    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

    # Validación visual y generación de información con el LLM (Gemini).
    # El LLM valida si la imagen corresponde a una hoja de caña de azúcar ("hoja_valida" -> "recognized").
    # Si hoja_valida es false, el LLM genera el campo "mensaje" explicando qué se ve y por qué no corresponde a caña.
    # Si hoja_valida es true, genera síntomas, causas y recomendaciones.
    disease_info = get_disease_info(result["class"], image_bytes, content_type)
    if "recognized" not in disease_info:
        disease_info["recognized"] = True

    # Si el LLM determinó que la imagen no corresponde a una hoja válida,
    # esto se refleja también en la clase mostrada al usuario.
    prediction = result
    if not disease_info.get("recognized", True):
        prediction = {**result, "class": "No identificada"}

    return {
        "filename": file.filename,
        "prediction": prediction,
        "disease_info": disease_info,
    }