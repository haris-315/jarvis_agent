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
    auth_token: str | None = None
    projects: list[str] | None = None
    tasks: list[dict] | None = None
    await websocket.accept()
    try:
        preProcessData = await websocket.receive_json()
        auth_token = preProcessData['authToken']
        projects = preProcessData['projects']
        tasks = preProcessData['tasks']
        print(f"Tasks: {tasks}\nProjects: {projects}\nAuthToken: {auth_token}")
        
        session_id = str(uuid.uuid4())
        session_memory[session_id] = {
            "auth_token": auth_token,
            "projects": projects,
            "tasks": tasks,
            "conversation": []
        }
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
                await process_transcript_streaming(websocket, session_id, transcript.text, session_memory)
            finally:
                is_processing = False
                await asyncio.sleep(0.06)

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