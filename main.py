import asyncio
import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import assemblyai as aai


load_dotenv()
api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not api_key:
    raise ValueError("ASSEMBLYAI_API_KEY is not set in the .env file")


aai.settings.api_key = api_key


print(f"AssemblyAI SDK version: {aai.__version__}")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def get():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    
    loop = asyncio.get_running_loop()

    
    try:
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=lambda transcript: asyncio.run_coroutine_threadsafe(
                websocket.send_text(json.dumps({"text": transcript.text if transcript.text else ""})), loop
            ),
            on_error=lambda error: print(f"AssemblyAI error: {error}"),
            on_open=lambda session: print(f"AssemblyAI session opened: {session}"),
            on_close=lambda: print("AssemblyAI session closed"),
        )
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        await websocket.close()
        return

    
    try:
        if transcriber is not None:
            transcriber.connect()
            print("Connected to AssemblyAI")
        else:
            print("Transcriber is None, cannot connect")
            await websocket.close()
            return
    except Exception as e:
        print(f"Failed to connect to AssemblyAI: {e}")
        await websocket.close()
        return

    try:
        while True:
            
            data = await websocket.receive_bytes()
            if not data:
                print("Received empty audio data")
                continue
            
            try:
                transcriber.stream(data)
            except Exception as e:
                print(f"Error streaming audio: {e}")
                continue
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        
        try:
            if transcriber is not None:
                await transcriber.close()
                print("Transcriber closed successfully")
            else:
                print("Transcriber is None, cannot close")
        except Exception as e:
            print(f"Error closing transcriber: {e}")
        await websocket.close()
        print("WebSocket connection closed")