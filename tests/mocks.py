"""
Módulo central de mocks compartidos para la suite de pruebas.

El objeto mock_db se crea aquí y se inyecta en app.database ANTES de
que cualquier módulo de la aplicación lo importe, evitando así el error
'Prisma object attribute is read-only'.
"""
from unittest.mock import AsyncMock, MagicMock

# Mock completo del cliente de Prisma
mock_db = MagicMock()
mock_db.connect = AsyncMock(return_value=None)
mock_db.disconnect = AsyncMock(return_value=None)
mock_db.user = MagicMock()
mock_db.user.find_unique = AsyncMock(return_value=None)
mock_db.user.create = AsyncMock(return_value=None)
