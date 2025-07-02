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
from typing import TypedDict
from openai import AsyncOpenAI
from datetime import datetime


load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not assemblyai_api_key:
    raise ValueError("ASSEMBLYAI_API_KEY is not set in the .env file")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")


aai.settings.api_key = assemblyai_api_key
openai_client = AsyncOpenAI(api_key=openai_api_key)


print(f"AssemblyAI SDK version: {aai.__version__}")


class AgentState(TypedDict):
    transcript: str
    response: str


async def process_transcript(state: AgentState) -> AgentState:
    transcript = state["transcript"]
    if not transcript:
        return {"transcript": transcript, "response": ""}
    
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
        return {"transcript": transcript, "response": f"Error processing transcript: {e}"}


workflow = StateGraph(AgentState)
workflow.add_node("process_transcript", process_transcript)
workflow.set_entry_point("process_transcript")
workflow.set_finish_point("process_transcript")
graph = workflow.compile()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def get():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connection established at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    
    loop = asyncio.get_running_loop()

    
    try:
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=lambda transcript: asyncio.run_coroutine_threadsafe(
                send_gpt_response(websocket, transcript.text, loop), loop
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
                transcriber.close()
                print("Transcriber closed successfully")
            else:
                print("Transcriber is None, cannot close")
        except Exception as e:
            print(f"Error closing transcriber: {e}")
        await websocket.close()
        print("WebSocket connection closed")


async def send_gpt_response(websocket: WebSocket, transcript: str, loop: asyncio.AbstractEventLoop):
    if not transcript:
        return
    try:
        
        result = await graph.ainvoke({"transcript": transcript, "response": ""})
        response = result["response"]
        if response and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"text": response}))
    except Exception as e:
        print(f"Error processing GPT response: {e}")