import pytest
import numpy as np
from PIL import Image
import io
from unittest.mock import patch, MagicMock
from app.services.model import preprocess_image, predict, CLASS_NAMES


def test_preprocess_image():
    # Arrange
    img = Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    image_bytes = img_byte_arr.getvalue()

    # Act
    result_tensor = preprocess_image(image_bytes)

    # Assert
    assert isinstance(result_tensor, np.ndarray)
    assert result_tensor.shape == (1, 224, 224, 3)  # ResNet50 input shape


@patch("app.services.model.load_model")
@patch("app.services.model.preprocess_image")
def test_predict_real_mode(mock_preprocess, mock_load_model):
    # Arrange
    mock_model = MagicMock()
    fake_prediction = np.array([[0.9] + [0.01] * 9])
    mock_model.predict.return_value = fake_prediction
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = np.zeros((1, 224, 224, 3))

    dummy_bytes = b"dummy"

    # Act
    result = predict(dummy_bytes)

    # Assert
    assert result["class"] == CLASS_NAMES[0]
    assert result["confidence"] == 0.9
    assert result["all_probabilities"][CLASS_NAMES[0]] == 0.9


# ===========================================================================
# CP-IA01 – Clasificación correcta de una imagen de cada tipo conocido
# ===========================================================================
@pytest.mark.parametrize("class_idx,expected_class", list(enumerate(CLASS_NAMES)))
@patch("app.services.model.load_model")
@patch("app.services.model.preprocess_image")
def test_predict_all_known_classes(mock_preprocess, mock_load_model, class_idx, expected_class):
    # Arrange
    mock_model = MagicMock()
    # Asignamos 0.91 a la clase evaluada y valores bajos (0.01) a las restantes
    probs = np.full((1, len(CLASS_NAMES)), 0.01, dtype=float)
    probs[0, class_idx] = 0.91
    mock_model.predict.return_value = probs
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = np.zeros((1, 224, 224, 3))

    dummy_bytes = b"dummy_image_data"

    # Act
    result = predict(dummy_bytes)

    # Assert
    assert result["class"] == expected_class
    assert result["confidence"] == 0.91
    assert len(result["all_probabilities"]) == len(CLASS_NAMES)
    assert result["all_probabilities"][expected_class] == 0.91


# ===========================================================================
# CP-IA02 – Comportamiento cuando el modelo no está disponible
# ===========================================================================
@patch("app.services.model.load_model")
def test_predict_model_not_available_raises_error(mock_load_model):
    # Arrange
    mock_load_model.return_value = None  # Sin modelo cargado

    # Act & Assert
    with pytest.raises(RuntimeError, match="El modelo de clasificación no está disponible"):
        predict(b"dummy")
