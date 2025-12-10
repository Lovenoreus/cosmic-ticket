# -------------------- Built-in Libraries --------------------
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
import uuid
from contextlib import asynccontextmanager

# -------------------- External Libraries --------------------
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# -------------------- LangGraph Dependencies --------------------
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

# -------------------- User-defined Modules --------------------
sys.path.insert(0, str(Path(__file__).parent / "tools"))
from tools.vector_database_tools import cosmic_database_tool2
import config
from utils import parse_json_from_response

load_dotenv(find_dotenv())

DEBUG = config.DEBUG if hasattr(config, 'DEBUG') else False

# ============================================================================
# PYDANTIC MODELS FOR MCP
# ============================================================================

from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: dict


class MCPToolsListResponse(BaseModel):
    tools: List[MCPTool]


class MCPContent(BaseModel):
    type: str
    text: str


class MCPToolCallRequest(BaseModel):
    name: str
    arguments: dict


class MCPToolCallResponse(BaseModel):
    content: List[MCPContent]
    isError: bool = False


# ============================================================================
# LANGGRAPH STATE AND DATACLASSES
# ============================================================================

@dataclass
class Question:
    """Represents a question in the ticket creation process"""
    question: str
    index: int
    answered: bool
    answer: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "index": self.index,
            "answered": self.answered,
            "answer": self.answer
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            question=data.get("question", ""),
            index=data.get("index", 0),
            answered=data.get("answered", False),
            answer=data.get("answer", None)
        )


class AgentState(TypedDict):
    user_input: str
    intent: dict
    conversation_history: List[BaseMessage]
    last_cosmic_query: Optional[str]
    last_cosmic_query_response: Optional[str]
    bot_response: Optional[str]
    known_problem_identified: bool
    questioning: bool
    questions: List[Dict[str, Any]]
    first_question_run_complete: bool


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

llm: ChatOpenAI = None
workflow_app = None
sessions: Dict[str, Dict[str, Any]] = {}
KNOWN_QUESTIONS = []


# ============================================================================
# LANGGRAPH AGENTS (same as your original code)
# ============================================================================

def detect_intent(state: AgentState) -> AgentState:
    """Detect user intent from input"""
    user_input = state["user_input"]
    conversation_history = state.get("conversation_history", [])
    questioning = state.get("questioning", False)

    if questioning:
        return {"intent": {"mode": "questioning"}, "conversation_history": conversation_history}

    system_prompt = """You are an intent detection agent. Analyze the user's message and determine their intent based on the conversation history.

Return ONLY a valid JSON object with one of these two structures:
- If the user is mentioning a problem or asking a question: {"mode": "cosmic_search"}
- If the user is talking about creating a support ticket: {"mode": "ticket_creation"}

Examples:
- "My printer is broken" -> {"mode": "cosmic_search"}
- "I need help with login" -> {"mode": "cosmic_search"}
- "How does <something> work?" -> {"mode": "cosmic_search"}
- "I want to create a ticket" -> {"mode": "ticket_creation"}

Return ONLY the JSON, nothing else."""

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    response_text = response.content.strip()
    intent = parse_json_from_response(response_text)

    updated_history = conversation_history + [
        HumanMessage(content=user_input),
        AIMessage(content=json.dumps(intent))
    ]

    return {"intent": intent, "conversation_history": updated_history}


async def cosmic_search_agent(state: AgentState) -> AgentState:
    """Cosmic search agent that handles cosmic_search mode"""
    user_input = state.get("user_input", "")
    last_cosmic_query = user_input

    tool_result = await cosmic_database_tool2(
        query=last_cosmic_query,
        collection_name=config.COSMIC_DATABASE_COLLECTION_NAME,
        limit=config.QDRANT_RESULT_LIMIT,
        min_score=config.QDRANT_MIN_SCORE
    )

    last_cosmic_query_response = json.dumps(tool_result, indent=2)
    bot_response = tool_result.get("message", "")

    if not bot_response:
        bot_response = 'I am sorry, I could not find any information on that topic.'

    updated_state = dict(state)
    updated_state["last_cosmic_query"] = last_cosmic_query
    updated_state["last_cosmic_query_response"] = last_cosmic_query_response
    updated_state["bot_response"] = bot_response

    return updated_state


def identify_known_question_agent(state: AgentState) -> AgentState:
    """Identify the most appropriate problem template from known_questions.json"""
    user_problem = state.get("last_cosmic_query", "") or state.get("user_input", "")

    if not user_problem:
        return {"known_problem_identified": False}

    templates_summary = []
    for idx, template in enumerate(KNOWN_QUESTIONS):
        template_info = {
            "index": idx,
            "issue_category": template.get("issue_category", ""),
            "description": template.get("description", ""),
            "keywords": template.get("keywords", []),
            "queue": template.get("queue", ""),
            "urgency_level": template.get("urgency_level", ""),
            "text": template.get("text", "")
        }
        templates_summary.append(template_info)

    system_prompt = """You are a problem classification agent. Match a user's problem to the most appropriate template.

Return format:
{
  "matched_index": <integer index>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}

If no good match (confidence < 0.5), return matched_index: -1"""

    templates_text = json.dumps(templates_summary, indent=2)
    user_prompt = f"""User Problem: {user_problem}

Available Templates: {templates_text}

Return ONLY the JSON response."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    try:
        match_result = parse_json_from_response(response_text, default={"matched_index": -1, "confidence": 0.0})
        matched_index = match_result.get("matched_index", -1)
        confidence = match_result.get("confidence", 0.0)

        if matched_index >= 0 and matched_index < len(KNOWN_QUESTIONS) and confidence >= 0.5:
            matched_template = KNOWN_QUESTIONS[matched_index]
            questions_to_ask = matched_template.get("questions_to_ask", [])
            questions = [
                Question(
                    question=q,
                    index=idx,
                    answered=False,
                    answer=None
                ).to_dict()
                for idx, q in enumerate(questions_to_ask, start=1)
            ]

            updated_state = dict(state)
            updated_state["known_problem_identified"] = True
            updated_state["matched_template"] = matched_template
            updated_state["match_confidence"] = confidence
            updated_state["questions"] = questions
            updated_state["questioning"] = True
            return updated_state
        else:
            updated_state = dict(state)
            updated_state["known_problem_identified"] = False
            updated_state["match_confidence"] = confidence
            return updated_state
    except Exception as e:
        updated_state = dict(state)
        updated_state["known_problem_identified"] = False
        updated_state["match_error"] = str(e)
        return updated_state


def questioner_agent(state: AgentState) -> AgentState:
    """Questioner agent that gathers answers for ticket creation"""
    questions_data = state.get("questions", [])
    user_input = state.get("user_input", "")
    conversation_history = state.get("conversation_history", [])
    first_question_run_complete = state.get("first_question_run_complete", False)

    if not questions_data:
        updated_state = dict(state)
        updated_state["bot_response"] = "Error: No questions initialized."
        updated_state["questioning"] = False
        return updated_state

    questions = [Question.from_dict(q) for q in questions_data]

    # Extract answers from user input (simplified version - you can expand this)
    if user_input:
        # Simple logic: mark first unanswered question as answered
        for q in questions:
            if not q.answered:
                q.answered = True
                q.answer = user_input
                break

    all_answered = all(q.answered for q in questions)
    unanswered_questions = [q for q in questions if not q.answered]

    if not first_question_run_complete:
        bot_response = "I need to gather some information:\n\n"
        for q in questions:
            bot_response += f"{q.index}. {q.question}\n"
        first_question_run_complete = True
    else:
        if all_answered:
            bot_response = "Thank you! I have all the information needed."
        else:
            bot_response = "Please answer:\n\n"
            for q in unanswered_questions:
                bot_response += f"{q.index}. {q.question}\n"

    questions_dicts = [q.to_dict() for q in questions]

    updated_state = dict(state)
    updated_state["questions"] = questions_dicts
    updated_state["questioning"] = not all_answered
    updated_state["bot_response"] = bot_response
    updated_state["first_question_run_complete"] = first_question_run_complete

    return updated_state


def route_after_intent(state: AgentState) -> str:
    """Route to appropriate agent based on detected intent"""
    intent = state.get("intent", {})
    mode = intent.get("mode", "cosmic_search")
    questioning = state.get("questioning", False)

    if questioning or mode == "questioning":
        return "questioner_agent"
    elif mode == "cosmic_search":
        return "cosmic_search_agent"
    elif mode == "ticket_creation":
        return "identify_known_question_agent"
    else:
        return END


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chatbot components before app starts"""
    global llm, workflow_app, KNOWN_QUESTIONS

    print("=" * 80)
    print("INITIALIZING COSMIC CHATBOT MCP SERVER")
    print("=" * 80)

    # Initialize LLM
    print("[STARTUP] Initializing LLM...")
    llm = ChatOpenAI(
        model=os.getenv("AGENT_MODEL_NAME", "gpt-4o-mini"),
        temperature=0
    )

    # Load known questions
    print("[STARTUP] Loading known questions...")
    KNOWN_QUESTIONS_PATH = Path(__file__).parent / "known_questions.json"
    if KNOWN_QUESTIONS_PATH.exists():
        with open(KNOWN_QUESTIONS_PATH, "r", encoding="utf8") as f:
            KNOWN_QUESTIONS = json.load(f)
        print(f"[STARTUP] Loaded {len(KNOWN_QUESTIONS)} known question templates")
    else:
        print("[STARTUP] Warning: known_questions.json not found")
        KNOWN_QUESTIONS = []

    # Build LangGraph workflow
    print("[STARTUP] Building LangGraph workflow...")
    workflow = StateGraph(AgentState)
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("cosmic_search_agent", cosmic_search_agent)
    workflow.add_node("identify_known_question_agent", identify_known_question_agent)
    workflow.add_node("questioner_agent", questioner_agent)

    workflow.set_entry_point("detect_intent")
    workflow.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "cosmic_search_agent": "cosmic_search_agent",
            "identify_known_question_agent": "identify_known_question_agent",
            "questioner_agent": "questioner_agent",
            END: END
        }
    )
    workflow.add_edge("cosmic_search_agent", END)
    workflow.add_conditional_edges(
        "identify_known_question_agent",
        lambda state: "questioner_agent" if state.get("questioning", False) else END,
        {
            "questioner_agent": "questioner_agent",
            END: END
        }
    )
    workflow.add_edge("questioner_agent", END)

    workflow_app = workflow.compile()
    print("[STARTUP] LangGraph workflow compiled successfully")

    print("=" * 80)
    print("COSMIC CHATBOT MCP SERVER READY")
    print("=" * 80)

    yield

    print("\n[SHUTDOWN] Cleaning up...")
    print("[SHUTDOWN] Server stopped")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Cosmic Chatbot MCP Server",
    description="MCP Server for Cosmic Knowledge Base and Support Ticket Creation",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Get existing session or create a new one"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_session_id = session_id or str(uuid.uuid4())
    sessions[new_session_id] = {
        "conversation_history": [],
        "last_cosmic_query": None,
        "last_cosmic_query_response": None,
        "known_problem_identified": False,
        "questioning": False,
        "questions": [],
        "first_question_run_complete": False
    }
    return new_session_id, sessions[new_session_id]


# ============================================================================
# MCP ENDPOINTS
# ============================================================================

@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list(request: Request):
    """MCP Protocol: List available tools"""
    query_params = request.query_params
    tools_filter = query_params.get("tools", None)

    if DEBUG:
        print(f"[MCP] tools/list called with filter: {tools_filter}")

    all_tools = [
        MCPTool(
            name="cosmic_chat",
            description="""
                TRIGGER: Any user question or support request

                ACTIONS:
                - Searches cosmic knowledge base for relevant information
                - Identifies known problems and initiates ticket creation workflow
                - Gathers required information through conversational questions

                SCOPE: IT support, troubleshooting, and ticket management

                RETURNS: Contextual response with relevant information or follow-up questions
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "User's question or support request"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to maintain conversation context"
                    }
                },
                "required": ["user_input"]
            }
        )
    ]

    if tools_filter:
        filtered_tools = [tool for tool in all_tools if tool.name == tools_filter]
        return MCPToolsListResponse(tools=filtered_tools)

    return MCPToolsListResponse(tools=all_tools)


@app.post("/mcp/tools/call", response_model=MCPToolCallResponse)
async def mcp_tools_call(request: MCPToolCallRequest):
    """MCP Protocol: Call a specific tool"""
    try:
        tool_name = request.name
        arguments = request.arguments

        if DEBUG:
            print(f"[MCP] Calling tool: {tool_name} with args: {arguments}")

        if tool_name == "cosmic_chat":
            user_input = arguments.get("user_input", "")
            session_id = arguments.get("session_id", None)

            if not user_input:
                return MCPToolCallResponse(
                    content=[MCPContent(type="text", text="Error: user_input is required")],
                    isError=True
                )

            # Get or create session
            session_id, session_state = get_or_create_session(session_id)

            # Prepare state
            current_state = {
                "user_input": user_input,
                "conversation_history": session_state.get("conversation_history", []),
                "last_cosmic_query": session_state.get("last_cosmic_query"),
                "last_cosmic_query_response": session_state.get("last_cosmic_query_response"),
                "bot_response": None,
                "known_problem_identified": session_state.get("known_problem_identified", False),
                "questioning": session_state.get("questioning", False),
                "questions": session_state.get("questions", []),
                "first_question_run_complete": session_state.get("first_question_run_complete", False)
            }

            if "matched_template" in session_state:
                current_state["matched_template"] = session_state["matched_template"]

            # Run workflow
            result = await workflow_app.ainvoke(current_state)

            # Update session
            sessions[session_id] = {
                "conversation_history": result.get("conversation_history", []),
                "last_cosmic_query": result.get("last_cosmic_query"),
                "last_cosmic_query_response": result.get("last_cosmic_query_response"),
                "known_problem_identified": result.get("known_problem_identified", False),
                "questioning": result.get("questioning", False),
                "questions": result.get("questions", []),
                "first_question_run_complete": result.get("first_question_run_complete", False)
            }

            if "matched_template" in result:
                sessions[session_id]["matched_template"] = result["matched_template"]

            # Prepare response
            response_data = {
                "bot_response": result.get("bot_response", "I'm not sure how to respond."),
                "session_id": session_id,
                "intent": result.get("intent"),
                "questioning": result.get("questioning", False),
                "questions": result.get("questions"),
                "metadata": {
                    "matched_template": result.get("matched_template", {}).get("issue_category") if result.get(
                        "matched_template") else None,
                    "match_confidence": result.get("match_confidence")
                }
            }

            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(response_data, indent=2))]
            )

        else:
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=f"Unknown tool: {tool_name}")],
                isError=True
            )

    except Exception as e:
        if DEBUG:
            print(f"[MCP] Error calling tool {request.name}: {e}")

        return MCPToolCallResponse(
            content=[MCPContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


@app.post("/")
async def mcp_streamable_http_endpoint(request: Request):
    """Streamable HTTP MCP protocol endpoint"""
    try:
        body = await request.json()
        method = body.get("method")
        request_id = body.get("id")

        if DEBUG:
            print(f"[STREAMABLE HTTP] Method: {method}")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "Cosmic Chatbot MCP Server",
                        "version": "1.0.0",
                        "description": "AI-powered support chatbot with cosmic knowledge base and ticket creation"
                    }
                }
            }

        elif method == "notifications/initialized":
            if DEBUG:
                print(f"[STREAMABLE HTTP] Connection established")
            return Response(status_code=204)

        elif method == "tools/list":
            params = body.get("params", {})
            tools_filter = params.get("tools", None)

            class MockRequest:
                def __init__(self, tools_filter):
                    self.query_params = {"tools": tools_filter} if tools_filter else {}

            mock_request = MockRequest(tools_filter)
            tools_response = await mcp_tools_list(mock_request)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [tool.model_dump() for tool in tools_response.tools]}
            }

        elif method == "tools/call":
            params = body.get("params", {})
            call_request = MCPToolCallRequest(**params)
            result = await mcp_tools_call(call_request)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result.model_dump()
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {}
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    except Exception as e:
        if DEBUG:
            print(f"[STREAMABLE HTTP] Error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


# ============================================================================
# STANDARD REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Cosmic Chatbot MCP Server",
        "timestamp": datetime.now().isoformat(),
        "llm_initialized": llm is not None,
        "workflow_initialized": workflow_app is not None,
        "known_questions_loaded": len(KNOWN_QUESTIONS),
        "active_sessions": len(sessions)
    }


@app.get("/info")
async def server_info():
    """Server information"""
    return {
        "service": "Cosmic Chatbot MCP Server",
        "version": "1.0.0",
        "description": "AI-powered support chatbot with cosmic knowledge base and ticket creation",
        "protocols": ["REST API", "MCP (Model Context Protocol)"],
        "mcp_endpoints": {
            "streamable_http": "/",
            "tools_list": "/mcp/tools/list",
            "tools_call": "/mcp/tools/call"
        },
        "features": [
            "Cosmic Knowledge Base Search",
            "Intelligent Problem Classification",
            "Automated Ticket Creation Workflow",
            "Conversational Question Gathering",
            "Session Management",
            "MCP Protocol Support"
        ],
        "tools": ["cosmic_chat"],
        "mcp_compatible": True
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cosmic Chatbot MCP Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(f"Starting Cosmic Chatbot MCP Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)