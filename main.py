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
from typing import TypedDict, List, Dict
from openai import AsyncOpenAI
from datetime import datetime
import uuid

load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not assemblyai_api_key or not openai_api_key:
    raise ValueError("API keys not properly set in .env")

aai.settings.api_key = assemblyai_api_key
openai_client = AsyncOpenAI(api_key=openai_api_key)

# Session-based memory store
session_memory: Dict[str, List[Dict[str, str]]] = {}

class AgentState(TypedDict):
    session_id: str
    transcript: str
    response: str

# -----------------------
# LangGraph agent logic
# -----------------------
async def process_transcript(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    transcript = state["transcript"]
    
    if not transcript or len(transcript.strip()) <= 5:
        return {"session_id": session_id, "transcript": "", "response": ""}
    
    try:
        # Retrieve session history or initialize it
        if session_id not in session_memory:
            session_memory[session_id] = []
        
        # Build message history for context
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond concisely to the user's input, considering prior conversation context."}
        ]
        # Add previous conversation history
        for entry in session_memory[session_id]:
            messages.append({"role": "user", "content": entry["transcript"]})
            messages.append({"role": "assistant", "content": entry["response"]})
        # Add current transcript
        messages.append({"role": "user", "content": transcript})

        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150
        )
        response = completion.choices[0].message.content
        
        # Store the interaction in session memory
        session_memory[session_id].append({"transcript": transcript, "response": response})
        
        # Limit memory size to prevent unbounded growth (e.g., last 10 interactions)
        if len(session_memory[session_id]) > 10:
            session_memory[session_id] = session_memory[session_id][-10:]
        
        return {"session_id": session_id, "transcript": transcript, "response": response}
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {"session_id": session_id, "transcript": transcript, "response": f"Error: {e}"}

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
    session_id = str(uuid.uuid4())  # Unique session ID for this WebSocket
    print(f"WebSocket connected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with session ID: {session_id}")

    loop = asyncio.get_running_loop()
    processing_lock = asyncio.Lock()
    last_transcript = None
    debounce_timeout = 1.0  # 1-second debounce for transcript processing

    async def on_data(transcript: aai.RealtimeTranscript):
        nonlocal last_transcript
        if not transcript.text or str(transcript.message_type) == "RealtimeMessageTypes.partial_transcript":
            return  # Skip partial transcripts
        
        # Debounce: only process if transcript is new and enough time has passed
        if transcript.text == last_transcript:
            return  # Skip duplicate transcripts
        last_transcript = transcript.text

        async with processing_lock:
            await send_gpt_response(websocket, session_id, transcript.text)
            await asyncio.sleep(debounce_timeout)  # Wait to avoid rapid successive calls

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
        # Clean up session memory
        if session_id in session_memory:
            del session_memory[session_id]
        await websocket.close()
        print(f"WebSocket closed for session ID: {session_id}")

# -----------------------
# Send response to frontend
# -----------------------
async def send_gpt_response(websocket: WebSocket, session_id: str, transcript: str):
    try:
        result = await graph.ainvoke({"session_id": session_id, "transcript": transcript, "response": ""})
        response = result["response"]
        if response and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"text": response}))
    except Exception as e:
        print(f"Error in GPT response: {e}")