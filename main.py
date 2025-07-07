import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from websocket_handler import websocket_endpoint, get_active_sessions

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Jarvis Task Manager", version="1.0.0")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_root(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Jarvis Task Manager is running",
        "sessions": get_active_sessions()
    }

@app.get("/sessions")
async def get_sessions():
    """Get active sessions info for debugging."""
    return get_active_sessions()

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")