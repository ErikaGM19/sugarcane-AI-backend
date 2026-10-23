import json
import pytest
from unittest.mock import patch, MagicMock
from app.services.llm import build_prompt, get_disease_info, DISEASE_DISPLAY_NAMES


# ===========================================================================
# CP-LLM01 – Generación correcta de síntomas/causas/recomendaciones
# ===========================================================================

def test_build_prompt_basic():
    # Arrange
    disease_name = "Carbón"

    # Act
    prompt = build_prompt(disease_name)

    # Assert
    assert "Carbón" in prompt
    assert "sintomas" in prompt
    assert "causas" in prompt
    assert "recomendaciones" in prompt
    assert "Responde en español" in prompt
    assert "Evita nombres científicos y tecnicismos" in prompt


def test_build_prompt_with_f1_score():
    # Arrange
    disease_name = "Carbón"
    f1 = 1.0

    # Act
    prompt = build_prompt(disease_name, f1_score=f1)

    # Assert
    assert "F1-Score del modelo de clasificación" in prompt
    assert "1.0000" in prompt


def test_build_prompt_all_classes():
    # Arrange & Act & Assert
    for internal_name, display_name in DISEASE_DISPLAY_NAMES.items():
        prompt = build_prompt(display_name)
        assert display_name in prompt
        assert "sintomas" in prompt
        assert "causas" in prompt
        assert "recomendaciones" in prompt

    # Verificar que el prompt contempla explícitamente el caso de hoja sana
    hoja_sana_prompt = build_prompt("Hoja sana")
    assert "Si la condición corresponde a una hoja sana" in hoja_sana_prompt


@patch("app.services.llm.get_client")
def test_get_disease_info_display_name_mapping(mock_get_client):
    # Arrange
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        '{"sintomas": "Tallo enrojecido con manchas blancas transversales.", '
        '"causas": "Infección fúngica favorecida por alta humedad.", '
        '"recomendaciones": "Usar variedades resistentes y desinfectar machetes.", '
        '"hoja_valida": true, "mensaje": ""}'
    )
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Act - Usamos el nombre interno del modelo ("Pudricion Roja")
    result = get_disease_info("Pudricion Roja", b"fake_image_bytes", "image/jpeg")

    # Assert
    # Debe mapear a "Muermo rojo" en el prompt enviado al LLM
    call_args = mock_client.models.generate_content.call_args
    prompt_sent = (
        call_args.kwargs["contents"][0]
        if isinstance(call_args.kwargs["contents"], list)
        else call_args.kwargs["contents"]
    )
    assert "Muermo rojo" in prompt_sent
    assert result["generated"] is True
    assert result["recognized"] is True
    assert "Tallo enrojecido" in result["sintomas"]
    assert "Infección fúngica" in result["causas"]
    assert "Usar variedades resistentes" in result["recomendaciones"]


# ===========================================================================
# CP-LLM02 – Comportamiento de respaldo con clave de API inválida
# ===========================================================================

@patch("app.services.llm.get_client")
def test_get_disease_info_failure(mock_get_client):
    # Arrange
    mock_get_client.side_effect = Exception("API Key Missing or network error")

    # Act
    result = get_disease_info("Carbon", b"fake_image_bytes", "image/jpeg")

    # Assert
    assert result["generated"] is False
    assert result["recognized"] is True
    assert result["mensaje"] == ""
    assert "No fue posible" in result["sintomas"]
    assert "No fue posible" in result["causas"]
    assert "Por favor consulta" in result["recomendaciones"]


# ===========================================================================
# CP-LLM03 – Formato JSON válido y robustez ante respuestas malformadas
# ===========================================================================

@patch("app.services.llm.get_client")
def test_get_disease_info_success(mock_get_client):
    # Arrange
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"sintomas": "Sintoma X", "causas": "Causa Y", "recomendaciones": "Recomendacion Z", "hoja_valida": true, "mensaje": ""}'
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Act
    result = get_disease_info("Carbon", b"fake_image_bytes", "image/jpeg")

    # Assert
    assert result["generated"] is True
    assert result["recognized"] is True
    assert result["sintomas"] == "Sintoma X"
    assert result["causas"] == "Causa Y"
    assert result["recomendaciones"] == "Recomendacion Z"
    assert result["mensaje"] == ""
    assert isinstance(result["sintomas"], str)
    assert isinstance(result["causas"], str)
    assert isinstance(result["recomendaciones"], str)
    assert isinstance(result["generated"], bool)
    assert isinstance(result["recognized"], bool)


@patch("app.services.llm.get_client")
def test_get_disease_info_malformed_json(mock_get_client):
    # Arrange
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Simula que Gemini devolvió texto plano o markdown roto en vez de JSON válido
    mock_response.text = "Error del modelo: no se generó un JSON válido {sin_cerrar"
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Act
    result = get_disease_info("Roya", b"fake_image_bytes", "image/jpeg")

    # Assert: debe activar el fallback seguro sin propagar json.JSONDecodeError
    assert result["generated"] is False
    assert result["recognized"] is True
    assert result["mensaje"] == ""
    assert "No fue posible generar información adicional" in result["sintomas"]
    assert isinstance(result["sintomas"], str)
    assert isinstance(result["causas"], str)
    assert isinstance(result["recomendaciones"], str)
    assert isinstance(result["generated"], bool)


@patch("app.services.llm.get_client")
def test_get_disease_info_schema_contract(mock_get_client):
    # Arrange: Verificar contrato de campos en éxito y fallo
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"sintomas": "S", "causas": "C", "recomendaciones": "R", "hoja_valida": true, "mensaje": ""}'
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Act
    success_result = get_disease_info("Hoja Sana", b"fake_image_bytes", "image/jpeg")

    mock_get_client.side_effect = RuntimeError("Conexión perdida")
    failure_result = get_disease_info("Hoja Sana", b"fake_image_bytes", "image/jpeg")

    # Assert
    expected_keys = {"sintomas", "causas", "recomendaciones", "mensaje", "generated", "recognized"}
    assert set(success_result.keys()) == expected_keys
    assert set(failure_result.keys()) == expected_keys


@patch("app.services.llm.get_client")
def test_get_disease_info_out_of_domain_image(mock_get_client):
    # Arrange: Simular que el LLM identifica que la imagen no es una hoja de caña
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "hoja_valida": False,
        "mensaje": "La imagen muestra un zapato y no corresponde a una hoja de caña de azúcar.",
        "sintomas": "",
        "causas": "",
        "recomendaciones": "",
    })
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Act
    result = get_disease_info("Hoja Amarilla", b"fake_shoe_bytes", "image/jpeg")

    # Assert
    assert result["generated"] is True
    assert result["recognized"] is False
    assert "no corresponde a una hoja de caña de azúcar" in result["mensaje"]
    assert result["sintomas"] == ""
    assert result["causas"] == ""
    assert result["recomendaciones"] == ""

