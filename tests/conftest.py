import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Importamos el mock_db centralizado e inyectamos en app.database ANTES
# de importar cualquier módulo de la app. Prisma no permite parchear
# sus atributos directamente (son read-only), así que reemplazamos el
# objeto completo en el módulo de base de datos.
# ---------------------------------------------------------------------------
from tests.mocks import mock_db

import app.database as db_module
db_module.db = mock_db

# Ahora es seguro importar la app
from app.main import app
from app import main as app_main_module

# También reemplazamos la referencia a 'db' que usa app.main directamente
app_main_module.db = mock_db


@pytest.fixture
def client():
    """
    Fixture que provee un TestClient con las dependencias pesadas mockeadas:
    - Base de datos Prisma → MagicMock (sin conexión real)
    - Modelo ML (TensorFlow) → no se carga en startup
    """
    with patch("app.main.load_model", return_value=None):
        with TestClient(app) as test_client:
            yield test_client
