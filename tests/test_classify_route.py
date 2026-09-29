import pytest
import io
import time
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image

from app.main import app
from app.dependencies import get_current_user
from tests.mocks import mock_db


def create_test_image(format="JPEG"):
    img = Image.new("RGB", (10, 10), color="green")
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# ===========================================================================
# CP-B02 – Endpoint de clasificación con imagen válida
# ===========================================================================

@patch("app.routes.classify.predict")
@patch("app.routes.classify.get_disease_info")
def test_classify_valid_image(mock_get_disease_info, mock_predict, client):
    # Arrange
    app.dependency_overrides[get_current_user] = lambda: MagicMock(email="user@test.com")

    mock_predict.return_value = {
        "class": "Hoja Sana",
        "confidence": 0.97,
        "all_probabilities": {},
    }
    mock_get_disease_info.return_value = {
        "sintomas": "Ninguno",
        "causas": "Hoja sana",
        "recomendaciones": "Mantener el monitoreo",
        "generated": True,
        "recognized": True,
    }

    files = {"file": ("hoja.jpg", create_test_image(), "image/jpeg")}

    # Act
    response = client.post("/classify", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "hoja.jpg"
    assert data["prediction"]["class"] == "Hoja Sana"
    assert data["disease_info"]["generated"] is True
    mock_predict.assert_called_once()
    mock_get_disease_info.assert_called_once_with("Hoja Sana")

    app.dependency_overrides.clear()


# ===========================================================================
# CP-B03 – Rechazo sin token o con token expirado/inválido
# ===========================================================================

def test_classify_no_token(client):
    # Arrange
    app.dependency_overrides.clear()
    files = {"file": ("hoja.jpg", create_test_image(), "image/jpeg")}

    # Act
    response = client.post("/classify", files=files)

    # Assert
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_classify_invalid_token(client):
    # Arrange
    app.dependency_overrides.clear()
    files = {"file": ("hoja.jpg", create_test_image(), "image/jpeg")}
    headers = {"Authorization": "Bearer token_invalido_xyz"}

    # Act
    response = client.post("/classify", files=files, headers=headers)

    # Assert
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"


# ===========================================================================
# CP-IA02 – Comportamiento ante error en el servicio de predicción
# ===========================================================================

@patch("app.routes.classify.predict")
def test_classify_model_error_route(mock_predict, client):
    # Arrange
    app.dependency_overrides[get_current_user] = lambda: MagicMock(email="user@test.com")
    mock_predict.side_effect = RuntimeError("El modelo de clasificación no está disponible o no se pudo cargar.")

    files = {"file": ("hoja.jpg", create_test_image(), "image/jpeg")}

    # Act
    response = client.post("/classify", files=files)

    # Assert
    assert response.status_code == 500
    assert "Error al procesar la imagen" in response.json()["detail"]

    app.dependency_overrides.clear()


# ===========================================================================
# CP-IA03 – Tiempo de respuesta razonable
# ===========================================================================

@patch("app.routes.classify.predict")
@patch("app.routes.classify.get_disease_info")
def test_classify_response_time(mock_get_disease_info, mock_predict, client):
    # Arrange
    app.dependency_overrides[get_current_user] = lambda: MagicMock(email="user@test.com")

    mock_predict.return_value = {
        "class": "Hoja Sana",
        "confidence": 0.98,
        "all_probabilities": {},
    }
    mock_get_disease_info.return_value = {
        "sintomas": "Sin síntomas",
        "causas": "Cultivo sano",
        "recomendaciones": "Monitoreo continuo",
        "generated": True,
    }

    files = {"file": ("hoja_tiempo.jpg", create_test_image(), "image/jpeg")}

    # Act
    start_time = time.perf_counter()
    response = client.post("/classify", files=files)
    elapsed_time = time.perf_counter() - start_time

    # Assert
    assert response.status_code == 200
    assert elapsed_time < 2.0  # El procesamiento debe completarse en menos de 2.0 segundos

    app.dependency_overrides.clear()


# ===========================================================================
# CP-IA04 – Robustez ante imágenes fuera del dominio (Baja confianza < 0.90)
# ===========================================================================

@patch("app.routes.classify.predict")
@patch("app.routes.classify.get_disease_info")
def test_classify_out_of_domain_image_low_confidence(mock_get_disease_info, mock_predict, client):
    # Arrange
    app.dependency_overrides[get_current_user] = lambda: MagicMock(email="user@test.com")

    # Simulamos una imagen fuera de dominio que genera predicción dispersa/baja confianza
    mock_predict.return_value = {
        "class": "Hoja Amarilla",
        "confidence": 0.45,  # Menor que CONFIDENCE_THRESHOLD (0.90)
        "all_probabilities": {"Hoja Amarilla": 0.45, "Carbon": 0.25},
    }

    files = {"file": ("zapato.jpg", create_test_image(), "image/jpeg")}

    # Act
    response = client.post("/classify", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["class"] == "No identificada"
    assert data["prediction"]["confidence"] == 0.45
    assert data["disease_info"]["recognized"] is False
    assert data["disease_info"]["generated"] is False
    assert "No fue posible asociar la imagen" in data["disease_info"]["sintomas"]
    # Verifica que NO se llamó al LLM innecesariamente cuando la confianza es insuficiente
    mock_get_disease_info.assert_not_called()

    app.dependency_overrides.clear()


# ===========================================================================
# CP-LLM02 – Respaldo en la ruta ante fallo o API key inválida del LLM
# ===========================================================================

@patch("app.routes.classify.predict")
@patch("app.routes.classify.get_disease_info")
def test_classify_route_llm_fallback_on_api_error(mock_get_disease_info, mock_predict, client):
    # Arrange
    app.dependency_overrides[get_current_user] = lambda: MagicMock(email="user@test.com")

    mock_predict.return_value = {
        "class": "Carbon",
        "confidence": 0.95,
        "all_probabilities": {},
    }
    # Simulamos que get_disease_info retorna fallback debido a API key inválida o desconexión
    mock_get_disease_info.return_value = {
        "sintomas": "No fue posible generar información adicional en este momento.",
        "causas": "No fue posible generar información adicional en este momento.",
        "recomendaciones": "Por favor consulta con un experto agrónomo o intenta nuevamente más tarde.",
        "generated": False,
    }

    files = {"file": ("carbon.jpg", create_test_image(), "image/jpeg")}

    # Act
    response = client.post("/classify", files=files)

    # Assert
    assert response.status_code == 200  # La ruta sigue funcionando sin fallar con error 500
    data = response.json()
    assert data["prediction"]["class"] == "Carbon"
    assert data["prediction"]["confidence"] == 0.95
    assert data["disease_info"]["generated"] is False
    assert data["disease_info"]["recognized"] is True

    app.dependency_overrides.clear()
