# -------------------- Built-in Libraries --------------------
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

# -------------------- External Libraries --------------------
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------- User-defined Modules --------------------
from bot import process_message, reset_conversation

load_dotenv(find_dotenv())

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
# PYDANTIC MODELS
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
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Cosmic Ticket Bot API",
    description="FastAPI endpoint for the Cosmic Ticket Bot - IT support ticket creation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
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
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Cosmic Ticket Bot API",
        "version": "1.0.0",
        "description": "FastAPI endpoint for the Cosmic Ticket Bot",
        "endpoints": {
            "send_message": "POST /message",
            "reset_session": "DELETE /session/{chat_id}",
            "health": "GET /health",
            "docs": "GET /docs",
            "openapi": "GET /openapi.json"
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
    logger.info("Docs: http://localhost:8500/docs")
    logger.info("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8500, log_level="info")

