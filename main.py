import asyncio
import json
import os
from fastapi import FastAPI, Request, WebSocket
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
import time

load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not assemblyai_api_key or not openai_api_key:
    raise ValueError("API keys not properly set in .env")

aai.settings.api_key = assemblyai_api_key
openai_client = AsyncOpenAI(api_key=openai_api_key)


session_memory: Dict[str, List[Dict[str, str]]] = {}

class AgentState(TypedDict):
    session_id: str
    transcript: str
    response: str

async def process_transcript_streaming(state: AgentState, websocket: WebSocket) -> AgentState:
    """Process transcript with direct streaming to websocket"""
    session_id = state["session_id"]
    transcript = state["transcript"]
    
    if not transcript or len(transcript.strip()) <= 5:
        return {"session_id": session_id, "transcript": "", "response": ""}
    
    try:
        if session_id not in session_memory:
            session_memory[session_id] = []
        
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond concisely to the user's input, considering prior conversation context."}
        ]
        
        
        for entry in session_memory[session_id]:
            messages.extend([
                {"role": "user", "content": entry["transcript"]},
                {"role": "assistant", "content": entry["response"]}
            ])
        messages.append({"role": "user", "content": transcript})

        
        stream = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            stream=True
        )
        
        
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                
                
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "text": content
                        }))
                    except Exception as e:
                        print(f"Error sending chunk: {e}")
                        break
        
        
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps({
                    "type": "end",
                    "text": ""
                }))
            except Exception as e:
                print(f"Error sending end marker: {e}")
        
        
        session_memory[session_id].append({"transcript": transcript, "response": full_response})
        
        
        if len(session_memory[session_id]) > 10:
            session_memory[session_id] = session_memory[session_id][-10:]
        
        return {"session_id": session_id, "transcript": transcript, "response": full_response}
        
    except Exception as e:
        print(f"OpenAI error: {e}")
        error_msg = f"Error: {e}"
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "text": error_msg
                }))
            except:
                pass
        return {"session_id": session_id, "transcript": transcript, "response": error_msg}


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get(request: Request):
    
    
    return templates.TemplateResponse("index.html",context={"request" : request})



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"WebSocket connected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with session ID: {session_id}")

    loop = asyncio.get_running_loop()
    processing_lock = asyncio.Lock()
    last_transcript = None
    debounce_timeout = 0.5  
    
    
    is_processing = False

    async def on_data(transcript: aai.RealtimeTranscript):
        nonlocal last_transcript, is_processing
        
        if not transcript.text or str(transcript.message_type) == "RealtimeMessageTypes.partial_transcript":
            return
        
        if transcript.text == last_transcript or is_processing:
            return
            
        last_transcript = transcript.text
        is_processing = True

        
        start_time = time.time()

        try:
            
            await send_gpt_response_streaming(websocket, session_id, transcript.text, start_time)
        finally:
            is_processing = False
            
            await asyncio.sleep(0.1)

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
        if session_id in session_memory:
            del session_memory[session_id]
        await websocket.close()
        print(f"WebSocket closed for session ID: {session_id}")

async def send_gpt_response_streaming(websocket: WebSocket, session_id: str, transcript: str, start_time: float):
    """Send GPT response with immediate streaming"""
    try:
        
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "start",
                "text": "",
                "transcript": transcript
            }))
        
        
        state = {"session_id": session_id, "transcript": transcript, "response": ""}
        result = await process_transcript_streaming(state, websocket)
        
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Processing time for transcript '{transcript[:30]}...': {total_time:.3f} seconds")
        
    except Exception as e:
        print(f"Error in GPT response: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "text": f"Error: {e}"
                }))
            except:
                pass