# -------------------- Built-in Libraries --------------------
import json
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -------------------- External Libraries --------------------
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------- User-defined Modules --------------------
from bot import process_message, reset_conversation

load_dotenv(find_dotenv())

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Thread pool executor for running sync bot functions in async context
executor = ThreadPoolExecutor(max_workers=10)


# ============================================================================
# PYDANTIC MODELS - REST API
# ============================================================================

class MessageRequest(BaseModel):
    """Request model for sending a message to the bot"""
    user_message: str = Field(..., description="The user's message to process")
    chat_id: str = Field(..., description="Conversation ID to maintain context across messages")

    class Config:
        json_schema_extra = {
            "example": {
                "user_message": "My printer is not working",
                "chat_id": "conversation_abc123"
            }
        }


class MessageResponse(BaseModel):
    """Response model from the bot"""
    success: bool
    message: str
    ticket_mode: bool
    ticket_ready: bool
    questions: Optional[list] = None
    answered_questions: Optional[list] = None
    unanswered_questions: Optional[list] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Based on the cosmic knowledge base...",
                "ticket_mode": False,
                "ticket_ready": False
            }
        }


class ResetResponse(BaseModel):
    """Response model for resetting a conversation"""
    success: bool
    message: str


# ============================================================================
# PYDANTIC MODELS - MCP
# ============================================================================

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
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server components"""
    logger.info("=" * 80)
    logger.info("INITIALIZING COSMIC CHATBOT SERVER")
    logger.info("=" * 80)
    logger.info("[STARTUP] Bot functionality loaded from bot.py")
    logger.info("[STARTUP] REST API endpoints available")
    logger.info("[STARTUP] MCP Protocol endpoints available")
    logger.info("=" * 80)
    logger.info("COSMIC CHATBOT SERVER READY")
    logger.info("=" * 80)

    yield

    logger.info("\n[SHUTDOWN] Cleaning up...")
    logger.info("[SHUTDOWN] Server stopped")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def process_message_async(chat_id: str, user_message: str) -> dict:
    """
    Async wrapper for process_message from bot.py
    Runs the sync function in a thread pool executor to avoid event loop conflicts
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_message, chat_id, user_message)
    return result


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Cosmic Ticket Bot API",
    description="FastAPI endpoint for the Cosmic Ticket Bot - IT support ticket creation system with MCP support",
    version="1.0.0",
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
# REST API ENDPOINTS
# ============================================================================

@app.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """
    Send a message to the bot and get a response.

    This endpoint processes user messages through the bot's conversation system.
    It maintains conversation context using the chat_id.

    - **user_message**: The user's message/question
    - **chat_id**: Unique identifier for the conversation thread (use the same chat_id for all messages in a conversation)

    Returns the bot's response along with ticket creation status if applicable.
    """
    logger.info(f"Received message request - chat_id: {request.chat_id}, message length: {len(request.user_message)}")

    try:
        # Process the message using async wrapper to avoid event loop conflicts
        result = await process_message_async(request.chat_id, request.user_message)

        logger.info(f"Message processed - success: {result.get('success')}, ticket_mode: {result.get('ticket_mode')}")

        # Convert to response model
        response = MessageResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            ticket_mode=result.get("ticket_mode", False),
            ticket_ready=result.get("ticket_ready", False),
            questions=result.get("questions"),
            answered_questions=result.get("answered_questions"),
            unanswered_questions=result.get("unanswered_questions"),
            error=result.get("error")
        )

        logger.info(f"Full Response: {response}")

        return response

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the message: {str(e)}"
        )


@app.delete("/session/{chat_id}", response_model=ResetResponse)
async def reset_session(chat_id: str):
    """
    Reset a conversation session.

    This endpoint clears all conversation history and state for the given chat_id,
    effectively starting a new conversation.

    - **chat_id**: The conversation ID to reset
    """
    logger.info(f"Resetting session: {chat_id}")

    try:
        success = reset_conversation(chat_id)

        if success:
            return ResetResponse(
                success=True,
                message=f"Session {chat_id} reset successfully"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {chat_id} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while resetting the session: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Cosmic Ticket Bot API",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "bot_loaded": True,
        "mcp_compatible": True
    }


@app.get("/info")
async def server_info():
    """Server information"""
    return {
        "service": "Cosmic Ticket Bot API",
        "version": "1.0.0",
        "description": "AI-powered support chatbot with cosmic knowledge base and ticket creation",
        "protocols": ["REST API", "MCP (Model Context Protocol)", "Streamable HTTP"],
        "endpoints": {
            "rest_api": {
                "send_message": "POST /message",
                "reset_session": "DELETE /session/{chat_id}",
                "health": "GET /health",
                "info": "GET /info",
                "docs": "GET /docs",
                "openapi": "GET /openapi.json"
            },
            "mcp": {
                "streamable_http": "/",
                "tools_list": "POST /mcp/tools/list",
                "tools_call": "POST /mcp/tools/call"
            }
        },
        "features": [
            "Cosmic Knowledge Base Search",
            "Intelligent Problem Classification",
            "Automated Ticket Creation Workflow",
            "Conversational Question Gathering",
            "Jira Ticket Integration",
            "JSON Ticket Storage",
            "Session Management via chat_id",
            "MCP Protocol Support"
        ],
        "tools": ["support_ticket_bot"],
        "mcp_compatible": True,
        "docs": "/docs"
    }


# ============================================================================
# MCP PROTOCOL ENDPOINTS
# ============================================================================

@app.post("/mcp/tools/list", response_model=MCPToolsListResponse)
async def mcp_tools_list(request: Request):
    """MCP Protocol: List available tools"""
    query_params = request.query_params
    tools_filter = query_params.get("tools", None)

    if DEBUG:
        logger.debug(f"[MCP] tools/list called with filter: {tools_filter}")

    all_tools = [
        MCPTool(
            name="support_ticket_bot",
            description=(
                "SCOPE: Use this tool for natural conversation-based support ticket creation. This is an intelligent conversational bot that guides users through creating support tickets by asking relevant questions based on the issue type.\n\n"
                "TRIGGER: User wants to have a conversation about an issue, needs help creating a ticket through natural dialogue, wants to report a problem through chat\n\n"
                "ACTION: Engages in a natural conversation with the user to:\n"
                "  1. Understand the problem through dialogue\n"
                "  2. Automatically identify the issue category from known categories\n"
                "  3. Ask relevant diagnostic questions based on the issue type\n"
                "  4. Collect all necessary information through conversation\n"
                "  5. Create both JSON and Jira tickets when all information is collected\n\n"
                "FEATURES:\n"
                "  - Maintains conversation context across multiple messages\n"
                "  - Automatically switches from normal chat to ticket mode when an issue is detected\n"
                "  - Searches cosmic knowledge base for relevant information\n"
                "  - Asks questions in batches, preserving original question indices\n"
                "  - Handles corrections and updates gracefully\n"
                "  - Creates tickets automatically when all questions are answered\n\n"
                "USE CASES:\n"
                "  - User says 'I have a problem with...' - start conversation\n"
                "  - User wants to report an issue through natural conversation\n"
                "  - User needs guidance on what information to provide\n"
                "  - User wants to create a ticket but doesn't know all details upfront\n\n"
                "RETURNS: Assistant's conversational response, current ticket state, and status information"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "The user's message in the conversation (REQUIRED)"
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Conversation ID to maintain context across messages (REQUIRED)"
                    }
                },
                "required": ["user_message", "chat_id"]
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
        arguments = request.arguments or {}

        if DEBUG:
            logger.debug(f"[MCP] Calling tool: {tool_name} with args: {arguments}")

        if tool_name == "support_ticket_bot":
            user_message = arguments.get("user_message", "")
            chat_id = arguments.get("chat_id", "")

            if not user_message:
                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": "user_message is required",
                            "message": "Please provide the user's message to process"
                        }, indent=2)
                    )],
                    isError=True
                )

            if not chat_id:
                return MCPToolCallResponse(
                    content=[MCPContent(
                        type="text",
                        text=json.dumps({
                            "success": False,
                            "error": "chat_id is required",
                            "message": "Please provide the conversation_id to maintain context"
                        }, indent=2)
                    )],
                    isError=True
                )

            # Process message using bot.py's process_message function
            result = await process_message_async(chat_id, user_message)

            if DEBUG:
                logger.debug(
                    f"[MCP] Bot result: success={result.get('success')}, ticket_mode={result.get('ticket_mode')}, ticket_ready={result.get('ticket_ready')}")

            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))],
                isError=not result.get("success", False)
            )

        else:
            return MCPToolCallResponse(
                content=[MCPContent(type="text", text=f"Unknown tool: {tool_name}")],
                isError=True
            )

    except Exception as e:
        if DEBUG:
            import traceback
            logger.error(f"[MCP] Error calling tool {request.name}: {e}")
            traceback.print_exc()

        return MCPToolCallResponse(
            content=[MCPContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "message": "An unexpected error occurred while processing the conversation"
                }, indent=2)
            )],
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
            logger.debug(f"[STREAMABLE HTTP] Method: {method}")

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
                        "name": "Cosmic Ticket Bot API",
                        "version": "1.0.0",
                        "description": "AI-powered support chatbot with cosmic knowledge base and ticket creation"
                    }
                }
            }

        elif method == "notifications/initialized":
            if DEBUG:
                logger.debug(f"[STREAMABLE HTTP] Connection established")
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
            logger.error(f"[STREAMABLE HTTP] Error: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id if 'request_id' in locals() else None,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Cosmic Ticket Bot API",
        "version": "1.0.0",
        "description": "FastAPI endpoint for the Cosmic Ticket Bot with MCP support",
        "protocols": ["REST API", "MCP (Model Context Protocol)", "Streamable HTTP"],
        "endpoints": {
            "rest_api": {
                "send_message": "POST /message",
                "reset_session": "DELETE /session/{chat_id}",
                "health": "GET /health",
                "info": "GET /info",
                "docs": "GET /docs",
                "openapi": "GET /openapi.json"
            },
            "mcp": {
                "streamable_http": "POST / (JSON-RPC)",
                "tools_list": "POST /mcp/tools/list",
                "tools_call": "POST /mcp/tools/call"
            }
        },
        "docs": "/docs"
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting Cosmic Ticket Bot API Server")
    logger.info("=" * 80)
    logger.info("Port: 8500")
    logger.info("REST API Docs: http://localhost:8500/docs")
    logger.info("MCP Protocol: Enabled")
    logger.info("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="info")
