from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model import predict
from app.services.llm import get_disease_info

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 5


@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
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

    # Consultar el módulo LLM con la clase detectada
    disease_info = get_disease_info(result["class"])

    return {
        "filename": file.filename,
        "prediction": result,
        "disease_info": disease_info,
    }