from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.services.model import predict
from app.services.llm import get_disease_info
from app.dependencies import get_current_user

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 5

# Umbral mínimo de confianza para aceptar una predicción como válida.
# Por debajo de esto, se asume que la imagen probablemente no es una
# hoja de caña de azúcar (u otra condición no vista en entrenamiento).
CONFIDENCE_THRESHOLD = 0.90

# Información de respaldo cuando la confianza es demasiado baja.
# No se consulta a Gemini en este caso.
UNRECOGNIZED_INFO = {
    "sintomas": "No fue posible asociar la imagen con ninguna de las enfermedades conocidas con suficiente certeza.",
    "causas": "Esto puede deberse a que la imagen no corresponde a una hoja de caña de azúcar, o a que la fotografía tiene mala iluminación, enfoque o encuadre.",
    "recomendaciones": "Verifica que la imagen sea de una hoja de caña de azúcar, tomada con buena luz y ocupando la mayor parte del encuadre. Si el problema persiste, consulta con un experto agrónomo.",
    "generated": False,
    "recognized": False,
}


@router.post("/classify")
async def classify_image(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    # Validar tipo de archivo
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no permitido: {file.content_type}. Usa JPG, PNG o WEBP."
        )

    image_bytes = await file.read()

    # Validar tamaño
    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"La imagen supera el límite de {MAX_SIZE_MB}MB."
        )

    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

    # Si la confianza es muy baja, no se trata como una enfermedad válida
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        return {
            "filename": file.filename,
            "prediction": {**result, "class": "No identificada"},
            "disease_info": UNRECOGNIZED_INFO,
        }

    # Consultar el módulo LLM con la clase detectada
    disease_info = get_disease_info(result["class"])
    disease_info["recognized"] = True

    return {
        "filename": file.filename,
        "prediction": result,
        "disease_info": disease_info,
    }