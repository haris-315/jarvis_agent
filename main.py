import asyncio
import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import assemblyai as aai

from fastapi.websockets import WebSocketState
from langgraph.graph import StateGraph
from langchain.agents import initialize_agent
from typing import TypedDict
from openai import AsyncOpenAI
from datetime import datetime

load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not assemblyai_api_key or not openai_api_key:
    raise ValueError("API keys not properly set in .env")

aai.settings.api_key = assemblyai_api_key
openai_client = AsyncOpenAI(api_key=openai_api_key)

class AgentState(TypedDict):
    transcript: str
    response: str

# -----------------------
# LangGraph agent logic
# -----------------------
async def process_transcript(state: AgentState) -> AgentState:
    transcript = state["transcript"]
    if not transcript or len(transcript.strip()) <= 5:
        return {"transcript": "", "response": ""}
    
    try:
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond concisely to the user's input."},
                {"role": "user", "content": transcript}
            ],
            max_tokens=150
        )
        response = completion.choices[0].message.content
        return {"transcript": transcript, "response": response}
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {"transcript": transcript, "response": f"Error: {e}"}

workflow = StateGraph(AgentState)
workflow.add_node("process_transcript", process_transcript)
workflow.set_entry_point("process_transcript")
workflow.set_finish_point("process_transcript")
graph = workflow.compile()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -----------------------
# HTML route
# -----------------------
@app.get("/")
async def get():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

# -----------------------
# WebSocket endpoint
# -----------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    loop = asyncio.get_running_loop()
    processing_lock = asyncio.Lock()

    async def on_data(transcript: aai.RealtimeTranscript):
        print(str(transcript.message_type))
        if not transcript.text or str(transcript.message_type) == "RealtimeMessageTypes.partial_transcript":
            return  # Skip interim results
        
        # Use lock to prevent overlapping GPT calls
        if processing_lock.locked():
            print("Still processing previous request, skipping")
            return

        async with processing_lock:
            await send_gpt_response(websocket, transcript.text)

    # Initialize transcriber
    try:
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=lambda t: asyncio.run_coroutine_threadsafe(on_data(t), loop),
            on_error=lambda e: print(f"AssemblyAI error: {e}"),
            on_open=lambda s: print(f"Session started: {s}"),
            on_close=lambda: print("AssemblyAI session closed"),
        )
    except Exception as e:
        print(f"Transcriber init failed: {e}")
        await websocket.close()
        return

    try:
        transcriber.connect()
    except Exception as e:
        print(f"Transcriber connect failed: {e}")
        await websocket.close()
        return

    # Stream audio from client to AssemblyAI
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            transcriber.stream(data)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            transcriber.close()
        except Exception as e:
            print(f"Error closing transcriber: {e}")
        await websocket.close()
        print("WebSocket closed")

# -----------------------
# Send response to frontend
# -----------------------
async def send_gpt_response(websocket: WebSocket, transcript: str):
    try:
        result = await graph.ainvoke({"transcript": transcript, "response": ""})
        response = result["response"]
        if response and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"text": response}))
    except Exception as e:
        print(f"Error in GPT response: {e}")
