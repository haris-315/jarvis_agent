import asyncio
import json
import uuid
from datetime import datetime
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
import assemblyai as aai
from transcript_processor import process_transcript_streaming

session_memory = {}

async def websocket_endpoint(websocket: WebSocket):
    authToken: str | None = None
    projects: list[str] | None = None
    tasks: list[dict] | None = None
    await websocket.accept()
    preProcessData = await websocket.receive_json()
    authToken = preProcessData['authToken']
    projects = preProcessData['projects']
    tasks = preProcessData['tasks']
    print(f"Tasks: {tasks} \n Projects: {projects} \n authToken: {authToken}")
    session_id = str(uuid.uuid4())
    print(f"WebSocket connected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with session ID: {session_id}")

    loop = asyncio.get_running_loop()
    last_transcript = None
    is_processing = False

    async def on_data(transcript: aai.RealtimeTranscript):
        nonlocal last_transcript, is_processing
        
        if not transcript.text or str(transcript.message_type) == "RealtimeMessageTypes.partial_transcript":
            return
        
        if transcript.text == last_transcript or is_processing:
            return
            
        last_transcript = transcript.text
        is_processing = True

        try:
            await send_gpt_response_streaming(websocket, session_id, transcript.text)
        finally:
            is_processing = False
            await asyncio.sleep(0.06)

    try:
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=lambda t: asyncio.run_coroutine_threadsafe(on_data(t), loop),
            on_error=lambda e: print(f"AssemblyAI error: {e}"),
            on_open=lambda s: print(f"Session started: {s}"),
            on_close=lambda: print("AssemblyAI session closed"),
        )
        transcriber.connect()
        
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

async def send_gpt_response_streaming(websocket: WebSocket, session_id: str, transcript: str):
    import time
    import json
    start_time = time.time()
    
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({
                "type": "start",
                "text": "",
                "transcript": transcript
            }))
        
        state = {"session_id": session_id, "transcript": transcript, "response": ""}
        await process_transcript_streaming(state, websocket, session_memory)
        
        end_time = time.time()
        print(f"Processing time for transcript '{transcript[:30]}...': {end_time - start_time:.3f} seconds")
        
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