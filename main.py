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
        {"desc": "Click the 'E-transfer' tab", "selector": "#nav-transfer"},
        {"desc": "Enter the payee name", "selector": "#payee-name"},
        {"desc": "Enter the amount", "selector": "#amount"},
        {"desc": "Click the 'Send' button", "selector": "#send-button"}
    ]
}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    step_index = body.get("stepIndex", 0)
    intent = body.get("intent") or None
    print(type(intent))
    if intent == "unknown" or "null":
        intent = None

    print("BACK intent: ", intent)

    deployment_name = "gpt-4"

    # 1. Intent not yet identified → ask GPT to classify it
    if not intent:
        print("HERE")
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
        print("BACK reply: ", reply)

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

    # 2. Intent is known, we are on a step — respond while staying on that step
    step_flow = flows.get(intent, [])
    print("BACK step_flow:", step_flow)
    if step_index < len(step_flow):
        current_step = step_flow[step_index]
    else:
        return {
            "intent": intent,
            "step": {},
            "stepIndex": step_index,
            "botMessage": "You've completed the e-transfer flow. Let me know if you need anything else!"
        }


    # Let GPT respond naturally, but instruct it to stay on this step
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
