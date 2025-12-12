"""
title: FastAPI Support Ticket Bot Filter
version: 0.7
"""

from pydantic import BaseModel
import requests
import json


class Filter:
    class Valves(BaseModel):
        priority: int = 0
        fastapi_bot_url: str = "http://host.docker.internal:8500"
        chat_id: str = "123"
        debug: bool = True
        stream_speed: int = 1  # Characters to stream per chunk (1 = one char at a time)

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

        # Store bot responses per message AND keep latest response
        self.bot_responses = {}
        self.latest_response = None  # Fallback for message_id mismatch

        # Track streaming position for each message
        self.stream_positions = {}

        print("=" * 60)
        print("FASTAPI BOT FILTER INITIALIZED")
        print(f"FastAPI Bot URL: {self.valves.fastapi_bot_url}")
        print("=" * 60)

    def inlet(self, body: dict):
        """Intercept user message and send to FastAPI bot"""
        if self.valves.debug:
            print("\n" + "=" * 60)
            print("INLET CALLED")
            print("=" * 60)

        messages = body.get("messages", [])

        if not messages:
            return body

        # Get the last user message
        last_message = messages[-1]
        user_message = last_message.get("content", "")

        if self.valves.debug:
            print(f"User message: '{user_message}'")

        if not user_message:
            return body

        # Get message_id to track the response
        message_id = body.get("metadata", {}).get("message_id", "default")

        if self.valves.debug:
            print(f"Message ID: {message_id}")

        try:
            api_url = f"{self.valves.fastapi_bot_url}/message"

            payload = {
                "user_message": user_message,
                "chat_id": self.valves.chat_id,
            }

            if self.valves.debug:
                print(f"Calling FastAPI bot at: {api_url}")
                print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            if self.valves.debug:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text[:500]}...")

            if response.status_code == 200:
                result = response.json()

                if self.valves.debug:
                    print(f"Parsed response keys: {result.keys()}")

                # Get the bot's message from the response
                bot_message = result.get("message", "")

                if self.valves.debug:
                    print(f"Bot message (first 100 chars): {bot_message[:100]}...")

                if bot_message:
                    print(f"✓ Got bot response: {bot_message[:80]}...")

                    # Store the bot response with BOTH message_id AND as latest
                    self.bot_responses[message_id] = bot_message
                    self.latest_response = bot_message  # Fallback storage

                    # Initialize stream position for this message
                    self.stream_positions[message_id] = 0

                    if self.valves.debug:
                        print(f"✓ Stored response for message_id: {message_id}")
                        print(f"✓ Stored as latest_response (fallback)")
                        print(f"✓ Initialized stream position to 0")
                        print(f"Success: {result.get('success')}")
                        print(f"Ticket mode: {result.get('ticket_mode')}")
                        print(f"Ticket ready: {result.get('ticket_ready')}")

                    # Give the LLM a minimal instruction to avoid it generating content
                    last_message["content"] = (
                        "[SYSTEM: Output exactly: 'OK']"
                    )
                else:
                    error_msg = "Error: No message in bot response"
                    if self.valves.debug:
                        print(f"ERROR: {error_msg}")
                        print(f"Full response: {json.dumps(result, indent=2)}")
                    self.bot_responses[message_id] = error_msg
                    self.latest_response = error_msg
                    self.stream_positions[message_id] = 0
                    last_message["content"] = (
                        "[SYSTEM: Output exactly: 'OK']"
                    )
            else:
                error_msg = f"Error: FastAPI bot returned status {response.status_code}"
                print(f"HTTP ERROR: {error_msg}")
                print(f"Response body: {response.text}")
                self.bot_responses[message_id] = error_msg
                self.latest_response = error_msg
                self.stream_positions[message_id] = 0
                last_message["content"] = (
                    "[SYSTEM: Output exactly: 'OK']"
                )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()
            self.bot_responses[message_id] = error_msg
            self.latest_response = error_msg
            self.stream_positions[message_id] = 0
            last_message["content"] = (
                "[SYSTEM: Output exactly: 'OK']"
            )

        if self.valves.debug:
            print(f"Stored message_ids: {list(self.bot_responses.keys())}")
            print(f"Latest response available: {self.latest_response is not None}")
            print(f"Stream positions: {self.stream_positions}")
            print("=" * 60 + "\n")

        return body

    def stream(self, event: dict) -> dict:
        """Stream bot response character by character"""
        if self.valves.debug:
            print(f"STREAM EVENT: {event}")

        # Try to find which message_id this stream belongs to
        # We'll use the latest_response as indicator that we have a bot response to stream
        message_id = None
        bot_response = None

        # Find the message_id for this stream
        for mid in self.stream_positions.keys():
            if mid in self.bot_responses:
                message_id = mid

                bot_response = self.bot_responses[mid]

                break

        # Fallback: use latest_response if no message_id found
        if bot_response is None and self.latest_response is not None:
            bot_response = self.latest_response
            message_id = "default"

            if message_id not in self.stream_positions:
                self.stream_positions[message_id] = 0

        # Replace the LLM's streaming content with our bot response
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})

            if "content" in delta and bot_response and message_id:
                # Get current position
                position = self.stream_positions.get(message_id, 0)

                # Get the next chunk of characters
                chunk_size = self.valves.stream_speed
                next_chunk = bot_response[position:position + chunk_size]

                if next_chunk:
                    # Replace with our chunk
                    delta["content"] = next_chunk

                    # Update position
                    self.stream_positions[message_id] = position + len(next_chunk)

                    if self.valves.debug:
                        print(
                            f"✓ Streaming chunk: '{next_chunk}' (pos: {position} -> {self.stream_positions[message_id]})")
                else:
                    # No more content to stream
                    delta["content"] = ""

                    if self.valves.debug:
                        print(f"✓ Finished streaming for message_id: {message_id}")

            elif "content" in delta:
                # Suppress LLM content if we don't have a bot response yet
                delta["content"] = ""

        return event

    def outlet(self, body: dict):
        """Clean up after streaming is complete"""
        if self.valves.debug:
            print("\n" + "=" * 60)
            print("OUTLET CALLED")
            print("=" * 60)

        # Get message_id
        message_id = body.get("metadata", {}).get("message_id", "default")

        if self.valves.debug:
            print(f"Looking for message_id: {message_id}")
            print(f"Available message_ids: {list(self.bot_responses.keys())}")

        bot_response = None

        # Try to get response by message_id first
        if message_id in self.bot_responses:
            bot_response = self.bot_responses[message_id]

            if self.valves.debug:
                print(f"✓ Found bot response for message_id: {message_id}")

            # Clean up
            del self.bot_responses[message_id]

            if message_id in self.stream_positions:
                del self.stream_positions[message_id]

        # Fallback to latest response if message_id doesn't match
        elif self.latest_response is not None:
            bot_response = self.latest_response

            if self.valves.debug:
                print(f"⚠ Message ID mismatch - using latest_response as fallback")

            # Clean up
            self.latest_response = None

            if message_id in self.stream_positions:
                del self.stream_positions[message_id]

        else:
            if self.valves.debug:
                print("⚠ No bot response found")

        # Ensure the final message is the complete bot response
        if bot_response:
            if self.valves.debug:
                print(f"Response preview: {bot_response[:100]}...")

            messages = body.get("messages", [])

            if messages:
                # Replace the last assistant message with complete response
                messages[-1]["content"] = bot_response

            if self.valves.debug:
                print(f"✓ Set final message to complete bot response")

        if self.valves.debug:
            print("=" * 60 + "\n")

        return body
