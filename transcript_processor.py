import json
from typing import TypedDict, Dict, List
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from openai import AsyncOpenAI
from config import openai_client

class AgentState(TypedDict):
    session_id: str
    transcript: str
    response: str

async def process_transcript_streaming(state: AgentState, websocket: WebSocket, session_memory: Dict[str, List[Dict[str, str]]]) -> AgentState:
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