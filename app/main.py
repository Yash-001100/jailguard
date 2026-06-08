"""JailGuard API — FastAPI application entry point."""
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup so first request isn't slow
    from inference.predict import _load
    _load()
    yield
    # Shutdown: nothing to clean up for PyTorch model


app = FastAPI(
    title="JailGuard API",
    description="Multi-turn jailbreak detection powered by SW-DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers.analyze import router
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
