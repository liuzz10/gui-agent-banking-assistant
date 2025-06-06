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

identify_intent_prompt = '''
You are a helpful banking assistant.
Your job is to identify the user's goal (choose from: e_transfer, pay_bills, check_balance).
Ask questions if unclear, but when certain, reply only with the intent name like 'e_transfer'.
'''

check_transferee_prompt = '''
You are helping the user transfer money.
You are currently in the 'check_transferee' step. Your goal is to guide the user through selecting the intended recipient.

Instructions:
1. Ask the user whether the person they want to transfer to is already listed on this page.
2. Based on the user's response:
   - If the user indicates **yes**, ask them for the recipient's name. After the user input a name (such as Alex), return a response like this:
     {
       "selector": "#contact-alex",
       "botMessage": "Please click on Alex Chen."
     }

   - If the user indicates **no**, return:
     {
       "selector": "#add-contact-button",
       "botMessage": "Please click the 'Add New Contact' button to add the recipient."
     }

Respond only with a valid JSON object in the format shown above.
'''

enter_amount_prompt = '''
The user is on the page to etransfer money to a recipient. Your goal is to guide the user to look for “From Account” to choose which of the accounts they'd like to transfer money from. Then, enter the amount they want to send. Once they've done that, click on “Continue”. Keep your reply short and easy to understand. Do not exceed 4 sentences.
'''

confirm_transfer_prompt = '''
The user is on the last step to e-transfer money. Your goal is to guide the user to double check the information and click on 'Confirm' if want to continue, otherwise click 'Cancel' to cancel the transaction.  Keep your reply short and easy to understand. Do not exceed 2 sentences.
'''

flows = {
    "e_transfer": [
        {"name": "go_to_tab", "desc": "Clicked the 'E-transfer' tab", "immediate_reply": "Click the 'E-transfer' tab", "selector": "#nav-transfer"},
        {"name": "check_transferee", "desc": "", "immediate_reply": "", "prompt": check_transferee_prompt},
        {"name": "enter_amount", "desc": "Entered amount and clicked 'Continue'", "simple_prompt": enter_amount_prompt, "selector": "#from-account, #amount, #send-button"},
        {"name": "confirm_transfer", "desc": "Clicked 'Confirm'", "immediate_reply": "", "simple_prompt": confirm_transfer_prompt, "selector": "#confirm-button, #cancel-button"}
    ]
}

deployment_name = "gpt-35-turbo"

def model_call(prompt, messages=[]):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": (prompt)}] + messages
    )
    return response.choices[0].message.content.strip()

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

    # 1. Intent identification phase
    # TODO: Identify intent at any step (not necessarily starting from 0)
    if not intent:
        res = model_call(identify_intent_prompt, messages)
        print("INTENT RESPONSE:", res)

        if res in flows:
            step = flows[res][0]
            return {
                "intent": res,
                "selector": step["selector"],
                "botMessage": step["immediate_reply"]
            }
        else:
            return {
                "intent": "unknown",
                "selector": "",
                "botMessage": res
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
        print("reply (raw):", reply)

        try:
            reply_dict = json.loads(reply)
            print("reply (parsed):", reply_dict)

            selector = reply_dict.get("selector", "")
            botMessage = reply_dict.get("botMessage", "Sorry, I didn’t understand that.")
            
            return {
                "intent": intent,
                "selector": selector,
                "stepName": step_name,
                "botMessage": botMessage
            }

        except json.JSONDecodeError:
            print("Failed to parse reply as JSON. Using fallback.")
            return {
                "intent": intent,
                "selector": "",
                "stepName": step_name,
                "botMessage": reply
            }

    
    # 4. Default contextual prompt (step-only, no substate)
    default_prompt_template = """
You are helping the user complete the '{intent}' task.

The current step is: "{step_name}", "{simple_prompt}"

Instructions:
- If the user asks unrelated questions, answer them politely.
- Always remind them to complete the current step by clicking the correct button.
- Do not move to the next step until the current one is completed.
"""
    prompt_text = default_prompt_template.format(
        intent=intent,
        step_name=step_name,
        simple_prompt=current_step["simple_prompt"]
    )

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": prompt_text}] + messages
    )

    reply = response.choices[0].message.content.strip()
    print("Simple reply:", reply)

    selector = current_step.get("selector", None)

    return {
        "intent": intent,
        "selector": selector,
        "stepName": step_name,
        "botMessage": reply
    }
