import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import assemblyai as aai
from transcript_processor import process_transcript_streaming

session_memory: Dict[str, Dict[str, Any]] = {}

async def websocket_endpoint(websocket: WebSocket):
    session_id = None
    transcriber = None
    
    try:
        await websocket.accept()
        
        # Receive initial data
        preProcessData = await websocket.receive_json()
        auth_token = preProcessData.get('authToken')
        projects = preProcessData.get('projects', [])
        tasks = preProcessData.get('tasks', [])
        
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
            except Exception as e:
                print(f"Error processing transcript: {e}")
            finally:
                is_processing = False
                await asyncio.sleep(0.06)

        def on_error(error):
            print(f"AssemblyAI error: {error}")

        def on_open(session):
            print(f"AssemblyAI session started: {session}")

        def on_close():
            print("AssemblyAI session closed")

        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=lambda t: asyncio.run_coroutine_threadsafe(on_data(t), loop),
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )
        
        transcriber.connect()
        
        while True:
            try:
                data = await websocket.receive_bytes()
                if not data:
                    break
                transcriber.stream(data)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break
            
    except WebSocketDisconnect:
        print("WebSocket disconnected during setup")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if transcriber:
            try:
                transcriber.close()
            except Exception as e:
                print(f"Error closing transcriber: {e}")
        
        if session_id and session_id in session_memory:
            del session_memory[session_id]
        
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except Exception as e:
            print(f"Error closing websocket: {e}")
        
        print(f"WebSocket closed for session ID: {session_id}")