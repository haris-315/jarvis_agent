import json
from typing import TypedDict, Dict, List
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END
from config import openai_client
import httpx

class AgentState(TypedDict):
    session_id: str
    transcript: str
    response: str
    messages: List[Dict]
    session_memory: Dict[str, Dict]

@tool
async def create_task(
    content: str,
    description: str,
    priority: int,
    project_id: int,
    due_date: str | None = None,
    reminder_at: str | None = None
) -> dict:
    """Create a new task in the task manager."""
    state = AgentStateRegistry.get_state()
    async with httpx.AsyncClient() as client:
        base_url = "http://your-api-base-url"  # Replace with your actual API base URL
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        response = await client.post(
            f"{base_url}/todo/tasks/",
            json={
                "content": content,
                "description": description,
                "priority": priority,
                "project_id": project_id,
                "due_date": due_date,
                "reminder_at": reminder_at
            },
            headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        )
        if response.status_code == 200:
            task = response.json()
            state["session_memory"][state["session_id"]]["tasks"].append(task)
            return {"status": "success", "task_id": task.get("id")}
        return {"error": f"Task creation failed: HTTP {response.status_code}"}

@tool
async def update_task(
    id: str,
    content: str,
    description: str,
    is_completed: bool,
    priority: int,
    project_id: int,
    due_date: str | None = None,
    reminder_at: str | None = None
) -> dict:
    """Update an existing task."""
    state = AgentStateRegistry.get_state()
    async with httpx.AsyncClient() as client:
        base_url = "http://your-api-base-url"
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        response = await client.put(
            f"{base_url}/todo/tasks/{id}",
            json={
                "content": content,
                "description": description,
                "is_completed": is_completed,
                "priority": priority,
                "project_id": project_id,
                "due_date": due_date,
                "reminder_at": reminder_at
            },
            headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        )
        if response.status_code == 200:
            task = response.json()
            tasks = state["session_memory"][state["session_id"]]["tasks"]
            index = next((i for i, t in enumerate(tasks) if t["id"] == id), -1)
            if index >= 0:
                tasks[index] = task
            else:
                tasks.append(task)
            return {"status": "success"}
        return {"error": f"Task update failed: HTTP {response.status_code}"}

@tool
async def create_project(
    name: str,
    color: str,
    is_favorite: bool,
    view_style: str
) -> dict:
    """Create a new project."""
    state = AgentStateRegistry.get_state()
    async with httpx.AsyncClient() as client:
        base_url = "http://your-api-base-url"
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        response = await client.post(
            f"{base_url}/todo/projects/",
            json={
                "name": name,
                "color": color,
                "is_favorite": is_favorite,
                "view_style": view_style
            },
            headers={"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        )
        if response.status_code == 200:
            project = response.json()
            state["session_memory"][state["session_id"]]["projects"].append(project.get("name"))
            return {"status": "success", "project_id": project.get("id", 0)}
        return {"error": f"Project creation failed: HTTP {response.status_code}"}

@tool
async def get_current_tasks() -> dict:
    """Retrieve the current list of tasks for the user."""
    state = AgentStateRegistry.get_state()
    return {"status": "success", "tasks": state["session_memory"][state["session_id"]]["tasks"]}

@tool
async def get_current_projects() -> dict:
    """Retrieve the current list of projects for the user."""
    state = AgentStateRegistry.get_state()
    return {"status": "success", "projects": state["session_memory"][state["session_id"]]["projects"]}

# Registry to hold the current state for tool access
class AgentStateRegistry:
    _state: AgentState = None

    @classmethod
    def set_state(cls, state: AgentState):
        cls._state = state

    @classmethod
    def get_state(cls) -> AgentState:
        if cls._state is None:
            raise ValueError("Agent state not set")
        return cls._state

# Define tools and model
tools = [create_task, update_task, create_project, get_current_tasks, get_current_projects]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_client.api_key).bind_tools(tools)

# Define graph
async def call_model(state: AgentState) -> AgentState:
    system_prompt = f"""
    You are Jarvis, a helpful assistant for a task manager app. Respond concisely in plain text suitable for text-to-speech, avoiding JSON or action details. Use function calls for actions like creating tasks, updating tasks, creating projects, or fetching current tasks/projects.

    Current projects: {', '.join(state['session_memory'][state['session_id']]['projects'])}
    Current tasks: {', '.join([t['content'] for t in state['session_memory'][state['session_id']]['tasks']])}

    Rules:
    - Use create_task for new tasks, assigning to 'Inbox' (project_id=1) if no project matches.
    - Use update_task for task modifications.
    - Use create_project for new projects.
    - Use get_current_tasks or get_current_projects to fetch task/project info when asked.
    - For prompts requiring multiple actions (e.g., create project and tasks), execute functions in the correct order: create project first, then tasks with the new project's ID.
    - Respond in a friendly, conversational tone.
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"] + [{"role": "user", "content": state["transcript"]}]
    AgentStateRegistry.set_state(state)  # Set state for tool access
    response = await model.ainvoke(messages)
    print(response)
    state["messages"].append(response.to_dict())
    state["response"] = response.content if response.content else ""
    return state

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", tools_condition, {"agent": "agent", END: END})
graph.set_entry_point("agent")
app = graph.compile()

async def process_transcript_streaming(websocket: WebSocket, session_id: str, transcript: str, session_memory: Dict[str, Dict]) -> None:
    if not transcript or len(transcript.strip()) <= 5:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "end", "text": ""}))
        return

    try:
        state = {
            "session_id": session_id,
            "transcript": transcript,
            "response": "",
            "messages": session_memory.get(session_id, {}).get("conversation", []),
            "session_memory": session_memory
        }

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "start", "text": "", "transcript": transcript}))

        async for partial_state in app.astream(state):
            if partial_state.get("response") and partial_state["response"] != state["response"]:
                state["response"] = partial_state["response"]
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps({"type": "chunk", "text": partial_state["response"]}))

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "end", "text": ""}))

        session_memory[session_id]["conversation"] = state["messages"][-10:]
        
    except Exception as e:
        print(f"Error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "text": f"Error: {e}"}))