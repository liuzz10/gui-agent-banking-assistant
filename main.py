from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from dotenv import load_dotenv
import ast
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

check_transferee_prompt = """
You are helping the user transfer money.
You are currently in the 'check_transferee' step. Your goal is to guide the user through selecting the intended recipient.

Instructions:
1. Ask the user whether the person they want to transfer to is already listed on this page.
2. Based on the user's response:
   - If the user indicates **yes**, ask them for the recipient's name and return a response like this:
     {
       "selector": "",
       "botMessage": "Please click on the recipient’s name if it is shown on the page."
     }

   - If the user indicates **no**, return:
     {
       "selector": "#add-contact-button",
       "botMessage": "Please click the 'Add New Contact' button to add the recipient."
     }

Respond only with a valid JSON object in the format shown above.
"""

flows = {
    "e_transfer": [
        {"name": "go_to_tab", "desc": "Click the 'E-transfer' tab", "selector": "#nav-transfer"},
        {"name": "check_transferee", "desc": "", "prompt": check_transferee_prompt},
        {"name": "enter_amount", "desc": "Enter amount and click 'Send'", "selector": "#amount, #send-button"},
        {"name": "confirm_transfer", "desc": "Click 'Confirm'", "selector": "#confirm-button"}
    ]
}

system_msg = {
    "role": "system",
    "content": (
        "You are a helpful banking assistant. "
        "Your job is to identify the user's goal (choose from: e_transfer, pay_bill, check_balance). "
        "Ask questions if unclear, but when certain, reply only with the intent name like 'e_transfer'."
    )
}

@app.post("/chat")
async def chat(request: Request):
    import json

    body = await request.json()
    messages = body.get("messages", [])
    step_index = body.get("stepIndex", 0)
    intent = body.get("intent") or None
    step_name = body.get("stepName")
    steps = []
    

    if intent in ["unknown", "null", "", None]:
        intent = None

    print("BACK intent:", intent)
    deployment_name = "gpt-4"

    # 1. Intent identification phase
    if not intent:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[system_msg] + messages
        )

        reply = response.choices[0].message.content.strip()
        print("reply:", reply)

        if reply in flows:
            step = flows[reply][0]
            return {
                "intent": reply,
                "selector": step["selector"],
                "stepIndex": 0,
                "botMessage": step["desc"]
            }
        else:
            return {
                "intent": "unknown",
                "selector": "",
                "stepIndex": 0,
                "botMessage": reply
            }

    # 2. Get step and prompt
    # If stepName is provided, recalculate stepIndex
    steps = flows.get(intent, [])

    print("step_name", step_name)
    # If stepName is provided, use it to resolve stepIndex
    if step_name:
        for i, step in enumerate(steps):
            if step.get("name") == step_name:
                step_index = i
                break

    # Validate stepIndex and set current_step
    if 0 <= step_index < len(steps):
        current_step = steps[step_index]
    else:
        return {
            "intent": intent,
            "selector": "",
            "stepName": step_name,
            "botMessage": "You've completed the e-transfer flow. Let me know if you need anything else!"
        }

    prompt = current_step.get("prompt", None)
    print("current_step", current_step)
    print("prompt", prompt)

    # 3. Handle logic within a step
    if prompt:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "system", "content": prompt}] + messages
        )
        
        reply = response.choices[0].message.content.strip()
        print("reply", reply)
        reply = json.loads(reply)
        print("reply", reply)  # Output: value
        selector = reply.get("selector", None)
        botMessage = reply.get("botMessage", "Sorry I don't understand. Can you say again?")
        return {
            "intent": intent,
            "selector": selector,
            "stepName": step_name,
            "botMessage": botMessage
        }
    
    # 4. Default contextual prompt (step-only, no substate)
    default_prompt_template = """
You are helping the user complete the '{intent}' task.

The current step is: "{step_name}"

Instructions:
- If the user asks unrelated questions, answer them politely.
- Always remind them to complete the current step by clicking the correct button.
- Do not move to the next step until the current one is completed.
"""
    prompt_text = default_prompt_template.format(
        intent=intent,
        step_name=step_name
    )

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": prompt_text}] + messages
    )

    reply = response.choices[0].message.content.strip()
    print("General reply:", reply)

    selector = current_step.get("selector", None)

    return {
        "intent": intent,
        "selector": selector,
        "stepName": step_name,
        "botMessage": reply
    }
