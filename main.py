import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from websocket_handler import websocket_endpoint

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

app.websocket("/ws")(websocket_endpoint)