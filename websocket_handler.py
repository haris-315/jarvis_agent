import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import assemblyai as aai
from transcript_processor import process_transcript_streaming

# Global session memory
session_memory: Dict[str, Dict[str, Any]] = {}

async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time transcription and processing."""
    session_id = None
    transcriber = None
    
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Receive initial data from client
        try:
            preProcessData = await websocket.receive_json()
            auth_token = preProcessData.get('authToken')
            projects = preProcessData.get('projects', [])
            tasks = preProcessData.get('tasks', [])
            
            print(f"Received data - Tasks: {len(tasks)}, Projects: {len(projects)}, AuthToken: {'***' if auth_token else 'None'}")
            
        except Exception as e:
            print(f"Error receiving initial data: {e}")
            await websocket.close(code=1002, reason="Invalid initial data")
            return
        
        # Create session
        session_id = str(uuid.uuid4())
        session_memory[session_id] = {
            "auth_token": auth_token,
            "projects": projects,
            "tasks": tasks,
            "conversation": []
        }
        
        print(f"Session created with ID: {session_id}")

        # Setup transcription handling
        loop = asyncio.get_running_loop()
        last_transcript = None
        is_processing = False
        processing_lock = asyncio.Lock()

        async def on_data(transcript: aai.RealtimeTranscript):
            """Handle incoming transcript data."""
            nonlocal last_transcript, is_processing
            
            # Skip partial transcripts and empty text
            if (not transcript.text or 
                str(transcript.message_type) == "RealtimeMessageTypes.partial_transcript" or
                len(transcript.text.strip()) <= 3):
                return
            
            # Avoid duplicate processing
            if transcript.text == last_transcript:
                return
                
            async with processing_lock:
                if is_processing:
                    return
                    
                is_processing = True
                last_transcript = transcript.text

            try:
                print(f"Processing transcript: {transcript.text}")
                await process_transcript_streaming(websocket, session_id, transcript.text, session_memory)
            except Exception as e:
                print(f"Error processing transcript: {e}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "error", 
                            "text": f"Processing error: {str(e)}"
                        }))
                    except:
                        pass
            finally:
                is_processing = False
                # Small delay to prevent rapid-fire processing
                await asyncio.sleep(0.1)

        def on_error(error):
            print(f"AssemblyAI error: {error}")

        def on_open(session):
            print(f"AssemblyAI session opened: {session}")

        def on_close():
            print("AssemblyAI session closed")

        # Create and start transcriber
        try:
            transcriber = aai.RealtimeTranscriber(
                sample_rate=16000,
                on_data=lambda t: asyncio.run_coroutine_threadsafe(on_data(t), loop),
                on_error=on_error,
                on_open=on_open,
                on_close=on_close,
            )
            
            transcriber.connect()
            print("Transcriber connected successfully")
            
        except Exception as e:
            print(f"Error creating transcriber: {e}")
            await websocket.close(code=1011, reason="Transcriber setup failed")
            return
        
        # Main message loop
        try:
            while True:
                try:
                    # Check if websocket is still connected
                    if websocket.client_state != WebSocketState.CONNECTED:
                        print("WebSocket disconnected, breaking loop")
                        break
                    
                    # Receive audio data
                    data = await websocket.receive_bytes()
                    
                    if not data:
                        print("Received empty data, breaking loop")
                        break
                    
                    # Stream data to transcriber
                    if transcriber:
                        transcriber.stream(data)
                        
                except WebSocketDisconnect:
                    print("WebSocket disconnected by client")
                    break
                except Exception as e:
                    print(f"Error in message loop: {e}")
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            
    except WebSocketDisconnect:
        print("WebSocket disconnected during setup")
    except Exception as e:
        print(f"Unexpected WebSocket error: {e}")
    finally:
        # Cleanup resources
        print(f"Cleaning up session: {session_id}")
        
        # Close transcriber
        if transcriber:
            try:
                transcriber.close()
                print("Transcriber closed")
            except Exception as e:
                print(f"Error closing transcriber: {e}")
        
        # Remove session from memory
        if session_id and session_id in session_memory:
            del session_memory[session_id]
            print(f"Session memory cleared for: {session_id}")
        
        # Close websocket if still connected
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
                print("WebSocket closed")
        except Exception as e:
            print(f"Error closing websocket: {e}")
        
        print(f"Session cleanup completed for: {session_id}")

# Health check function for debugging
def get_active_sessions():
    """Get information about active sessions."""
    return {
        "active_sessions": len(session_memory),
        "session_ids": list(session_memory.keys())
    }