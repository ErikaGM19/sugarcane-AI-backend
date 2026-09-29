import pytest
from jose import jwt
from datetime import datetime, timedelta
from app.services.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS
)

def test_password_hashing():
    # Arrange
    password = "supersecretpassword"
    
    # Act
    hashed = get_password_hash(password)
    
    # Assert
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_create_access_token():
    # Arrange
    subject = "test@example.com"
    
    # Act
    token = create_access_token(subject)
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    # Assert
    assert token is not None
    assert decoded["sub"] == subject
    assert decoded["type"] == "access"
    assert "exp" in decoded

def test_create_refresh_token():
    # Arrange
    subject = "test@example.com"
    
    # Act
    token = create_refresh_token(subject)
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    # Assert
    assert token is not None
    assert decoded["sub"] == subject
    assert decoded["type"] == "refresh"
    assert "exp" in decoded
