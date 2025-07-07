import json
from typing import TypedDict, Dict, List, Any, Optional
from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from config import openai_client
import httpx

class AgentState(TypedDict):
    session_id: str
    transcript: str
    response: str
    messages: List[Any]
    session_memory: Dict[str, Dict[str, Any]]

@tool
async def create_task(
    content: str,
    description: str,
    priority: int,
    project_id: int,
    due_date: Optional[str] = None,
    reminder_at: Optional[str] = None
) -> dict:
    """Create a new task in the task manager."""
    state = AgentStateRegistry.get_state()
    async with httpx.AsyncClient() as client:
        base_url = "https://jarvis.trylenoxinstruments.com"  # Using the URL from your update_task function
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        try:
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
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

@tool
async def update_task(
    id: str,
    content: str,
    description: str,
    is_completed: bool,
    priority: int,
    project_id: int,
    due_date: Optional[str] = None,
    reminder_at: Optional[str] = None
) -> dict:
    """Update an existing task."""
    state = AgentStateRegistry.get_state()
    async with httpx.AsyncClient() as client:
        base_url = "https://jarvis.trylenoxinstruments.com"
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        try:
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
                index = next((i for i, t in enumerate(tasks) if t.get("id") == id), -1)
                if index >= 0:
                    tasks[index] = task
                else:
                    tasks.append(task)
                return {"status": "success"}
            return {"error": f"Task update failed: HTTP {response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

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
        base_url = "https://jarvis.trylenoxinstruments.com"
        auth_token = state["session_memory"][state["session_id"]]["auth_token"]
        try:
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
                state["session_memory"][state["session_id"]]["projects"].append(project.get("name", name))
                return {"status": "success", "project_id": project.get("id", 0)}
            return {"error": f"Project creation failed: HTTP {response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

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
model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_client.api_key).bind_tools(tools)

def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or end."""
    if not state["messages"]:
        return "end"
    
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

async def call_model(state: AgentState) -> AgentState:
    """Call the model with the current state."""
    try:
        # Get current session data
        session_data = state["session_memory"][state["session_id"]]
        
        system_prompt = f"""
You are Jarvis, a helpful assistant for a task manager app. Respond concisely in plain text suitable for text-to-speech, avoiding JSON or action details. Use function calls for actions like creating tasks, updating tasks, creating projects, or fetching current tasks/projects.

Current projects: {', '.join(session_data['projects'])}
Current tasks: {', '.join([str(t.get('content', 'Unknown')) for t in session_data['tasks']])}

Rules:
- Use create_task for new tasks, assigning to 'Inbox' (project_id=1) if no project matches.
- Use update_task for task modifications.
- Use create_project for new projects.
- Use get_current_tasks or get_current_projects to fetch task/project info when asked.
- For prompts requiring multiple actions (e.g., create project and tasks), execute functions in the correct order: create project first, then tasks with the new project's ID.
- Respond in a friendly, conversational tone.
"""

        # Build messages list
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history (excluding duplicate system messages)
        for msg in state["messages"]:
            if not isinstance(msg, SystemMessage):
                messages.append(msg)
        
        # Add current transcript if not already in messages
        if not any(isinstance(m, HumanMessage) and m.content == state["transcript"] for m in messages):
            messages.append(HumanMessage(content=state["transcript"]))

        # Set state for tools
        AgentStateRegistry.set_state(state)

        # Get response from model
        response = await model.ainvoke(messages)
        print(f"Model response: {response}")

        # Update state with response
        state["messages"] = messages[1:] + [response]  # Exclude system message from stored messages
        state["response"] = response.content if response.content else ""
        
        return state
        
    except Exception as e:
        print(f"Error in call_model: {e}")
        error_message = f"Sorry, I ran into an issue: {str(e)}. Please try again or ask for something else."
        state["messages"].append(AIMessage(content=error_message))
        state["response"] = error_message
        return state

async def custom_tool_node(state: AgentState) -> AgentState:
    """Custom tool node to handle tool execution and append ToolMessage."""
    if not state["messages"]:
        return state
        
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls"):
        return state
        
    tool_calls = last_message.tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Find and execute the tool
        tool_found = False
        for tool in tools:
            if tool.name == tool_name:  # Use .name instead of .__name__
                tool_found = True
                try:
                    result = await tool.ainvoke(tool_args)
                    tool_messages.append(ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tool_id,
                        name=tool_name
                    ))
                    
                    # Update response based on tool result
                    if result.get("status") == "success":
                        if tool_name == "create_project":
                            state["response"] = f"Project '{tool_args.get('name', 'Unknown')}' created successfully!"
                        elif tool_name == "create_task":
                            state["response"] = f"Task '{tool_args.get('content', 'Unknown')}' added successfully!"
                        elif tool_name == "update_task":
                            state["response"] = f"Task updated successfully!"
                        elif tool_name == "get_current_projects":
                            projects = result.get("projects", [])
                            state["response"] = f"Your current projects are: {', '.join(projects) if projects else 'none'}."
                        elif tool_name == "get_current_tasks":
                            tasks = [str(t.get("content", "Unknown")) for t in result.get("tasks", [])]
                            state["response"] = f"Your current tasks are: {', '.join(tasks) if tasks else 'none'}."
                    else:
                        error_msg = result.get("error", "Unknown error")
                        state["response"] = f"Sorry, I couldn't complete that action: {error_msg}. Please try again."
                        
                except Exception as e:
                    error_message = f"Tool {tool_name} failed: {str(e)}"
                    tool_messages.append(ToolMessage(
                        content=json.dumps({"error": error_message}),
                        tool_call_id=tool_id,
                        name=tool_name
                    ))
                    state["response"] = f"Sorry, I couldn't {tool_name.replace('_', ' ')}: {str(e)}. Please try again."
                break
        
        if not tool_found:
            error_message = f"Tool {tool_name} not found"
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": error_message}),
                tool_call_id=tool_id,
                name=tool_name
            ))
            state["response"] = f"Sorry, I couldn't find the requested function. Please try again."

    # Add tool messages to state
    state["messages"].extend(tool_messages)
    return state

# Build the graph
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", custom_tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    graph.add_edge("tools", "agent")
    return graph.compile()

# Create the compiled graph
app = build_graph()

async def process_transcript_streaming(websocket: WebSocket, session_id: str, transcript: str, session_memory: Dict[str, Dict[str, Any]]) -> None:
    """Process transcript and send streaming response."""
    # Skip empty or very short transcripts
    if not transcript or len(transcript.strip()) <= 3:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "end", "text": ""}))
        return

    try:
        # Get conversation history
        conversation_history = session_memory.get(session_id, {}).get("conversation", [])

        # Create initial state
        state = {
            "session_id": session_id,
            "transcript": transcript,
            "response": "",
            "messages": conversation_history.copy(),
            "session_memory": session_memory
        }

        # Send start message
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "start", "text": "", "transcript": transcript}))

        # Process the state through the graph
        result = await app.ainvoke(state)
        print(f"Graph result: {result}")

        # Send response if available
        if result.get("response") and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "chunk", "text": result["response"]}))

        # Send end message
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "end", "text": ""}))

        # Update session memory (limit conversation history to last 10 messages)
        filtered_messages = [m for m in result["messages"] if not isinstance(m, SystemMessage)]
        session_memory[session_id]["conversation"] = filtered_messages[-10:]

    except Exception as e:
        print(f"Error in process_transcript_streaming: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(json.dumps({"type": "error", "text": f"Error: {str(e)}. Please try again."}))