from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE")
)

flows = {
    "e_transfer": [
        {"name": "go_to_tab", "desc": "Click the 'E-transfer' tab", "selector": "#nav-transfer"},
        {"name": "check_transferee", "desc": "Is the person you want to transfer to listed on this page? If yes, select them. If not, click 'Add New Contact'.", "selector": ".contact-button"},
        {"name": "enter_amount", "desc": "Enter the amount and click the 'Send' button", "selector": "#amount, #send-button"},
        {"name": "confirm_transfer", "desc": "Click the 'Confirm' button to complete the transfer", "selector": "#confirm-button"}
    ]
}

def handle_step_override(intent: str, step_name: str, messages: list):
    if intent == "e_transfer" and step_name == "check_transferee":
        last_msg = messages[-1]["content"].lower().strip()

        if last_msg in ["no", "not here", "nope"]:
            return {
                "intent": intent,
                "step": {"name": "check_transferee"},
                "stepIndex": 1,
                "botMessage": "No problem. You can click 'Add New Contact'.",
                "extraInstruction": {"instruction": "highlightAddContact"}
            }

        elif last_msg in ["yes", "yep", "i see them", "yes they are here"]:
            return {
                "intent": intent,
                "step": {"name": "ask_recipient_name"},
                "stepIndex": 1.5,
                "botMessage": "Great. What is the recipient's name?"
            }

        elif step_name == "ask_recipient_name":
            recipient_name = messages[-1]["content"].strip()
            return {
                "intent": intent,
                "step": {"name": "await_contact_click"},
                "stepIndex": 1.6,
                "botMessage": f"Looking for {recipient_name}... Click the contact if it's highlighted.",
                "extraInstruction": {
                    "instruction": "highlightRecipientByName",
                    "name": recipient_name
                }
            }

    return None

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    step_index = body.get("stepIndex", 0)
    intent = body.get("intent") or None
    step_name = body.get("stepName")

    if intent in ["unknown", "null", "", None]:
        intent = None

    print("BACK intent:", intent)
    deployment_name = "gpt-4"

    # 1. Intent identification
    if not intent:
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful banking assistant. "
                "Your job is to identify the user's goal (choose from: e_transfer, pay_bill, check_balance). "
                "Ask questions if unclear, but when certain, reply only with the intent name like 'e_transfer'."
            )
        }

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[system_msg] + messages
        )

        reply = response.choices[0].message.content.strip()
        print("BACK reply:", reply)

        if reply in flows:
            step = flows[reply][0]
            return {
                "intent": reply,
                "step": step,
                "stepIndex": 0,
                "botMessage": step["desc"]
            }
        else:
            return {
                "intent": "unknown",
                "step": {},
                "stepIndex": 0,
                "botMessage": reply
            }

    # 2. Named step override if provided
    if step_name and intent in flows:
        for i, s in enumerate(flows[intent]):
            if s.get("name") == step_name:
                return {
                    "intent": intent,
                    "step": s,
                    "stepIndex": i,
                    "botMessage": s["desc"]
                }

    # 3. Normal step flow
    step_flow = flows.get(intent, [])
    if step_index < len(step_flow):
        current_step = step_flow[step_index]
    else:
        return {
            "intent": intent,
            "step": {},
            "stepIndex": step_index,
            "botMessage": "You've completed the e-transfer flow. Let me know if you need anything else!"
        }

    contextual_system_msg = {
        "role": "system",
        "content": (
            f"You are helping the user complete the '{intent}' task. "
            f"The current step is: \"{current_step['desc']}\".\n"
            f"If the user asks unrelated questions, answer them politely. "
            f"Always remind them to complete the current step by clicking the correct button.\n"
            f"Do not move to the next step until the current one is completed."
        )
    }

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[contextual_system_msg] + messages
    )

    bot_reply = response.choices[0].message.content.strip()

    return {
        "intent": intent,
        "step": current_step,
        "stepIndex": step_index,
        "botMessage": bot_reply
    }
