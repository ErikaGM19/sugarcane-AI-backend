from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import classify, chat, auth
from app.services.model import load_model
from app.database import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    load_model()
    yield
    # Shutdown
    await db.disconnect()


app = FastAPI(title="Sugarcane Disease API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(classify.router)
app.include_router(chat.router)