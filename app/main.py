from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import classify, chat, auth
from app.services.model import load_model
from app.database import db

app = FastAPI(title="Sugarcane Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await db.connect()
    load_model() 

@app.on_event("shutdown")
async def shutdown_event():
    await db.disconnect()

app.include_router(auth.router)
app.include_router(classify.router)
app.include_router(chat.router)