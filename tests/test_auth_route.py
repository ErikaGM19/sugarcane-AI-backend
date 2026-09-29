import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.auth import get_password_hash
from tests.mocks import mock_db


# Mock class para simular un usuario de la base de datos
class MockUser:
    def __init__(self, email, password_hash, is_active=True):
        from datetime import datetime
        self.id = "mock-user-id-001"
        self.email = email
        self.password_hash = password_hash
        self.is_active = is_active
        self.created_at = datetime(2024, 1, 1, 0, 0, 0)
        self.updated_at = datetime(2024, 1, 1, 0, 0, 0)



# ===========================================================================
# CP-B01 – Login con credenciales válidas e inválidas
# ===========================================================================

def test_login_success(client):
    # Arrange
    hashed_pw = get_password_hash("password123")
    mock_db.user.find_unique = AsyncMock(
        return_value=MockUser("test@example.com", hashed_pw)
    )

    # Act
    response = client.post("/auth/login", data={
        "username": "test@example.com",
        "password": "password123"
    })

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(client):
    # Arrange
    hashed_pw = get_password_hash("password123")
    mock_db.user.find_unique = AsyncMock(
        return_value=MockUser("test@example.com", hashed_pw)
    )

    # Act
    response = client.post("/auth/login", data={
        "username": "test@example.com",
        "password": "wrongpassword"
    })

    # Assert
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


# ===========================================================================
# Tests adicionales de autenticación
# ===========================================================================

def test_register_success(client):
    # Arrange
    mock_db.user.find_unique = AsyncMock(return_value=None)
    mock_db.user.create = AsyncMock(
        return_value=MockUser("nuevo@example.com", "hashed")
    )

    # Act
    response = client.post("/auth/register", json={
        "email": "nuevo@example.com",
        "password": "password123"
    })

    # Assert
    assert response.status_code == 201
    assert response.json()["email"] == "nuevo@example.com"


def test_register_existing_user(client):
    # Arrange
    mock_db.user.find_unique = AsyncMock(
        return_value=MockUser("test@example.com", "hashed")
    )

    # Act
    response = client.post("/auth/register", json={
        "email": "test@example.com",
        "password": "password123"
    })

    # Assert
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]
