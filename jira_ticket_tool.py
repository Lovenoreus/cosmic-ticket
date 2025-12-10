"""Jira Ticket ITSM System"""
# -------------------- Built-in Libraries --------------------
import os

# -------------------- External Libraries --------------------
import requests
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

print("Found env:", find_dotenv())


def create_jira_ticket(
        conversation_id: str = "",
        conversation_topic: str = "",
        description: str = "",
        location: str = "",
        queue: str = "",
        priority: str = "",
        department: str = "",
        name: str = "",
        category: str = ""
) -> dict:
    """
    Creates a support ticket in Jira using the provided details.

    Args:
        conversation_id (str): Thread identifier to get stored fields
        conversation_topic (str): Short summary of the issue.
        description (str): Detailed explanation of the issue (should include "Problem Analysis" section).
        location (str): Where the issue occurred.
        queue (str): Which team/department is responsible.
        priority (str): Priority level of the issue.
        department (str): The user's department.
        name (str): Name of the person reporting.
        category (str): Ticket category.

    Returns:
        dict: Response with ticket key and status.
    """

    # Use provided values or fall back to stored values
    final_values = {
        'conversation_topic': conversation_topic,
        'description': description,
        'location': location,
        'queue': queue,
        'priority': priority,
        'department': department,
        'name': name,
        'category': category
    }

    print(f'📩 Creating Jira ticket for topic: "{final_values["conversation_topic"]}"')

    # Load config
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
    JIRA_EMAIL = os.getenv("JIRA_EMAIL")
    JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")

    print(f'🔧 Jira config - Domain: {JIRA_DOMAIN}, Email: {JIRA_EMAIL}, Token Set: {bool(JIRA_API_TOKEN)}')

    if not all([JIRA_API_TOKEN, JIRA_EMAIL, JIRA_DOMAIN]):
        msg = "❌ Jira configuration is incomplete. Please set environment variables properly."
        print(msg)
        return {"success": False, "error": msg}

    PROJECT_KEY = "HEAL"
    ISSUE_TYPE = "Task"
    CREATE_ISSUE_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue"
    TRANSITION_URL = f"https://{JIRA_DOMAIN}/rest/api/3/issue/{{}}/transitions"

    HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # Format description
    content_blocks = []
    
    # Check if description is already formatted with fields (new format)
    if final_values['description'] and any(field in final_values['description'] for field in ["Summary:", "Assigned Queue:", "Priority:"]):
        # Parse the pre-formatted description
        desc_lines = final_values['description'].split("\n")
        current_label = None
        current_content = []
        in_conversation_history = False
        
        for line in desc_lines:
            stripped_line = line.strip()
            
            # Check if this line is a label (ends with colon, not User: or AI Assistant:)
            if stripped_line.endswith(":") and not stripped_line.startswith("User:") and not stripped_line.startswith("AI Assistant:"):
                # Flush previous content
                if current_label and current_content:
                    content_blocks.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": current_label, "marks": [{"type": "strong"}]}]
                    })
                    if in_conversation_history:
                        # For Conversation History, add each line separately
                        for hist_line in current_content:
                            if hist_line.strip():
                                content_blocks.append({
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": hist_line}]
                                })
                    else:
                        # For other fields, join content
                        content_text = "\n".join(current_content).strip()
                        if content_text:
                            content_blocks.append({
                                "type": "paragraph",
                                "content": [{"type": "text", "text": content_text}]
                            })
                
                current_label = stripped_line
                current_content = []
                in_conversation_history = "Conversation History" in current_label
            elif current_label:
                # This is content for the current label
                current_content.append(line.rstrip())
        
        # Flush remaining content after loop
        if current_label and current_content:
            content_blocks.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": current_label, "marks": [{"type": "strong"}]}]
            })
            if in_conversation_history:
                for hist_line in current_content:
                    if hist_line.strip():
                        content_blocks.append({
                            "type": "paragraph",
                            "content": [{"type": "text", "text": hist_line}]
                        })
            else:
                content_text = "\n".join(current_content).strip()
                if content_text:
                    content_blocks.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": content_text}]
                    })
    else:
        # Legacy format - use old parsing logic
        cleaned_description = final_values['description'].replace("Problem Analysis: ", "", 1) if final_values[
            'description'] else "No description provided"

        content_blocks.extend([
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Summary:", "marks": [{"type": "strong"}]}]
            },
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": cleaned_description}]
            }
        ])

        def add_block(label: str, value: str):
            if value:  # Only add if value exists
                content_blocks.extend([
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": f"{label}:", "marks": [{"type": "strong"}]}]
                    },
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": str(value)}]
                    }
                ])

        add_block("Location", final_values['location'])
        add_block("Assigned Queue", final_values['queue'])
        add_block("Priority", final_values['priority'])
        add_block("Department", final_values['department'])
        add_block("Name", final_values['name'])
        add_block("Category", final_values['category'])
        add_block("Conversation Topic", final_values['conversation_topic'])

    content_blocks.append({
        "type": "paragraph",
        "content": [{"type": "text", "text": "Call Ended", "marks": [{"type": "strong"}]}]
    })

    # Build labels - add escalation label if escalated
    labels = ["Tickets"]

    # Build the summary to include escalation info if present
    summary = final_values['conversation_topic']

    payload = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": content_blocks
            },
            "issuetype": {"name": ISSUE_TYPE},
            "labels": labels
        }
    }

    try:
        print(f"[DEBUG] Sending payload to Jira...")

        response = requests.post(
            CREATE_ISSUE_URL,
            headers=HEADERS,
            auth=(JIRA_EMAIL, JIRA_API_TOKEN),
            json=payload
        )

        print(f"[DEBUG] Jira response status: {response.status_code}")

        if response.status_code == 201:
            issue_key = response.json()["key"]
            print(f"✅ Ticket {issue_key} successfully created!")

            # Transition the ticket - fixed to use string ID
            try:
                transition_payload = {"transition": {"id": "51"}}
                transition_response = requests.post(
                    TRANSITION_URL.format(issue_key),
                    headers=HEADERS,
                    auth=(JIRA_EMAIL, JIRA_API_TOKEN),
                    json=transition_payload
                )

                if transition_response.status_code == 204:
                    print(f"✅ Ticket {issue_key} transitioned to 'Tickets'.")

                else:
                    print(
                        f"⚠️ Created ticket, but transition failed (status {transition_response.status_code}): {transition_response.text}")

            except Exception as transition_error:
                print(f"⚠️ Ticket created but transition failed: {str(transition_error)}")

            # Return structured response with full escalation info
            return {
                "success": True,
                "key": issue_key,
                "jira_key": issue_key,
                "message": f"✅ Ticket {issue_key} has been created successfully!",
                "original_reporter": final_values['name']
            }

        else:
            error_msg = response.json() if response.content else "Unknown error"
            print(f"❌ Failed to create ticket: {error_msg}")

            return {
                "success": False,
                "error": f"Failed to create ticket: {error_msg}"
            }

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")

        if 'response' in locals():
            print(f"❌ Response status: {response.status_code}")
            print(f"❌ Response content: {response.text}")

        return {
            "success": False,
            "error": f"Exception occurred: {str(e)}"
        }