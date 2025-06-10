from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from dotenv import load_dotenv
from collections import OrderedDict
import ast
import os
import json

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
Do not exceed 100 charaters in your reply. Do not exceed 2 sentences.
'''

identify_intent_prompt = '''
You are a helpful banking assistant. Your job is to identify the user's goal (choose one of: e_transfer, pay_bills, check_balance).
If the user’s intent is clear, reply with one of the three intent names only.
If the intent is unclear or ambiguous (e.g., user says “not sure”), ask a clarifying question. Do not guess. Do not assume. Do not reply with an intent unless you are certain from the user’s input.
Keep responses under 100 characters. Use only one or two sentences max.

'''

CLICK_ETRANSFER_BTN_PROMPT = '''
You are helping the user transfer money. Your job is to guide the user to click the "e-Transfer" tab on the top right of the website. The button is highlighted in yellow and labeled "e-Transfer".
Do not exceed 80 charaters or 1 sentence in your reply.
'''

check_transferee_prompt = '''
You are helping the user transfer money.
You are currently on the page of selecting a recipient. Your goal is to guide the user through selecting the intended recipient.
First, ask the user whether the person they want to transfer to is already listed on this page. Based on the user's response:
If the user indicates **yes**, ask them to click on the recipient.
If the user indicates **no**, ask them to click the 'Add New Contact' button to add the recipient.
Do not exceed 120 charaters or 2 sentences in your reply.
'''

ENTER_AMOUNT_PROMPT = '''
The user is on the page to etransfer money to a recipient. Your goal is to guide the user to look for “From Account” to choose which of the accounts they'd like to transfer money from. Then, enter the amount they want to send. Once they've done that, click on “Continue”. They can also click on "Cancel" at any time. Keep your reply short and easy to understand. Do not exceed 2 sentences.
'''

confirm_transfer_prompt = '''
The user is on the last step to e-transfer money. Your goal is to guide the user to double check the information and click on 'Confirm' if want to continue, otherwise click 'Cancel' to cancel the transaction.  Keep your reply short and easy to understand. Do not exceed 2 sentences.
'''

INTENT_PROMPT = '''
You are a helpful banking assistant. Your job is to identify the user's goal (choose one of: e_transfer, pay_bills, check_balance).
If the user's goal is clear, reply with exactly one of the intent names above.
If the user's goal is unclear, ambiguous, or missing, respond with exactly: clarification_required.
Do not guess. Do not explain your choice. Do not include any punctuation or extra words.
'''

CLARIFICATION_PROMPT = """
The user’s intent is unclear. Your job is to ask a short, polite follow-up question that will help determine whether the user wants to:
- e_transfer
- pay_bills
- check_balance

You must respond with **a single, clear question**, no longer than 20 words. Do not guess the user's intent. Only ask one question at a time.

Examples:
- Would you like to transfer money, pay a bill, or check your balance?
- What banking task would you like to complete today?
"""

e_transfer = OrderedDict({
    "index.html": {
        "immediate_reply": "Click the 'e-Transfer' tab on the top right of the page",
        "selector": "#nav-transfer",
        "prompt": CLICK_ETRANSFER_BTN_PROMPT,
        "desc": "Clicked the 'E-transfer' tab"

    },
    "etransfer.html": {
        "immediate_reply": "Please select the recipient you want to transfer money to.",
        "selector": "",
        "prompt": check_transferee_prompt,
        "desc": "Selected the recipient"
    },
    "send_to_alex.html": {
        "prompt": ENTER_AMOUNT_PROMPT,  # Optional: keep general one
        "substeps": OrderedDict({
            "choose_account": {
                "selector": "#from-account",
                "immediate_reply": "Please choose the account you want to transfer from.",
                "completion_condition": "account_chosen",  # flag name
                "desc": "Selected the account to transfer from"
            },
            "enter_amount": {
                "selector": "#amount",
                "immediate_reply": "Now enter the amount.",
                "completion_condition": "amount_entered",
                "desc": "Entered the amount to transfer"
            },
            "continue": {
                "selector": "#send-button",
                "immediate_reply": "Now click 'Continue'.",
                "completion_condition": "continue_clicked",
                "desc": "Clicked 'Continue'"
            },
        }),
        "desc": "Filled in information and clicked 'Continue'"
    },
    "confirm_transfer.html": {
        "immediate_reply": "Please double-check the information and click 'Confirm' to complete the transfer, or 'Cancel' if you want to stop.",
        "selector": "#confirm-button, #cancel-button",
        "prompt": confirm_transfer_prompt,
        "desc": "Clicked 'Confirm'"
    }
})

flows = {
    "e_transfer": e_transfer,
}

deployment_name = "gpt-35-turbo"

# Utility function to merge consecutive messages with the same role. GPT expects alternating roles.
def merge_consecutive_messages(messages):
    if not messages:
        return []

    merged = [messages[0]]
    for msg in messages[1:]:
        last = merged[-1]
        if msg["role"] == last["role"]:
            last["content"] += " " + msg["content"]
        else:
            merged.append(msg)
    return merged

def api_call(prompt, messages=[]):
    cleaned_messages = merge_consecutive_messages(messages)
    # print("API prompt", prompt)
    print("API cleaned_messages", cleaned_messages)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": (prompt)}] + cleaned_messages
    )
    print("API response:", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    new_page_loaded = body.get("newPageLoaded", False)
    subtask_updated = body.get("substepUpdated", False)
    intent = body.get("intent") or None
    substep_flags = body.get("substep_flags", {})   # example: {"account_chosen": True}
    current_page = body.get("currentPage")    # e.g., check_transferee
    current_step = {}   # e.g., {"desc": "Clicked 'Confirm'", "immediate_reply": "", "prompt": confirm_transfer_prompt, "selector": "#confirm-button, #cancel-button"}

    if intent in ["unknown", "null", "", "undefined", None]:
        intent = None

    print("====Intent and current_page from frontend:", intent, current_page)

    # 1. Intent Identification
    if not intent:
        # Ask questions until intent is identified
        print("====Identifying intent...")
        intent = api_call(INTENT_PROMPT, messages)
        if intent == "clarification_required":
            print("====Intent unclear, asking for clarification...")
            follow_up = api_call(CLARIFICATION_PROMPT, messages)
            return {
                "intent": "unknown",
                "selector": "",
                "botMessage": follow_up,
            }
        elif intent in flows:
            if current_page in flows[intent]:
                current_step = flows[intent][current_page]
                if "selector" in current_step:
                    print("====Current step found in flows:", current_step)
                    return {
                        "intent": intent,
                        "selector": current_step["selector"],
                        "botMessage": current_step["immediate_reply"]
                    }
                else:
                    print("====WIP: No selector found for current step in intent:", current_step)
            else:
                print("====current_page not in flows[res]")
        else:
            print("====WIP: Intent not found in flows (GPT hallucination or not built yet)", intent)
    # When an intent is identified:
    # If a new page just loaded, then send an initial instruction directly
    # If the user is on the page for a while, and has updated a subtask, then send the next substep instruction
    else:
        if new_page_loaded or subtask_updated:
            if intent in flows and current_page in flows[intent]:
                current_step = flows[intent][current_page]
                substeps = current_step.get("substeps", {})
                # If there are substeps, check their completion conditions
                # and send the message for the first uncompleted substep
                if substeps:
                    for _, substep in substeps.items():
                        condition = substep.get("completion_condition") # e.g., "account_chosen"
                        print("====Substep_flags:", substep_flags)
                        if not substep_flags.get(condition): # substep_flags looks like this: {"account_chosen": True} => {"account_chosen": True, "amount_entered": False}
                            return {
                                "intent": intent,
                                "selector": substep["selector"],
                                "botMessage": substep["immediate_reply"],
                                "substep_flags": substep_flags
                            }
                # If no substeps, proceed with the current step
                elif "selector" in current_step:
                    return {
                        "intent": intent,
                        "selector": current_step["selector"],
                        "botMessage": current_step["immediate_reply"]
                    }
                else:
                    print("====WIP: No selector found for current step in intent:", current_step)
            else:
                print("====WIP: Current page not found in flows for intent:", intent, current_page)
        else:
            # If intent is already identified && at least one instruction has been sent (user asks other questions after the next action has been highlighted), then answer their questions
            # Also need to track if the user change their intent or not here
            if intent in flows and current_page in flows[intent]:
                current_step = flows[intent][current_page]
                print("====current_step", current_step)
                res = api_call(current_step["prompt"], messages)
                return {
                    "botMessage": res
                }
    



    # 2. Get step and prompt    
    # # Validate stepIndex and set current_step
    # if 0 <= step_index < len(steps):
    #     current_step = steps[step_index]
    # else:
    #     return {
    #         "intent": intent,
    #         "selector": "",
    #         "botMessage": "You've completed the e-transfer flow. Let me know if you need anything else!"
    #     }

    # prompt = current_step.get("prompt", None)
    # print("current_step", current_step)
    # print("prompt", prompt)

    # # 3. Handle logic within a step
    # if prompt:
    #     response = client.chat.completions.create(
    #         model=deployment_name,
    #         messages=[{"role": "system", "content": prompt}] + messages
    #     )
        
    #     reply = response.choices[0].message.content.strip()
    #     print("reply (raw):", reply)

    #     try:
    #         reply_dict = json.loads(reply)
    #         print("reply (parsed):", reply_dict)

    #         selector = reply_dict.get("selector", "")
    #         botMessage = reply_dict.get("botMessage", "Sorry, I didn’t understand that.")
            
    #         return {
    #             "intent": intent,
    #             "selector": selector,
    #             # "stepName": step_name,
    #             "botMessage": botMessage
    #         }

    #     except json.JSONDecodeError:
    #         print("Failed to parse reply as JSON. Using fallback.")
    #         return {
    #             "intent": intent,
    #             "selector": "",
    #             # "stepName": step_name,
    #             "botMessage": reply
    #         }

    
#     # 4. Default contextual prompt (step-only, no substate)
#     default_prompt_template = """
# You are helping the user complete the '{intent}' task.

# The current step is: "{step_name}", "{prompt}"

# Instructions:
# - If the user asks unrelated questions, answer them politely.
# - Always remind them to complete the current step by clicking the correct button.
# - Do not move to the next step until the current one is completed.
# """
#     prompt_text = default_prompt_template.format(
#         intent=intent,
#         # step_name=step_name,
#         prompt=current_step["prompt"]
#     )

#     response = client.chat.completions.create(
#         model=deployment_name,
#         messages=[{"role": "system", "content": prompt_text}] + messages
#     )

#     reply = response.choices[0].message.content.strip()
#     print("Simple reply:", reply)

#     selector = current_step.get("selector", None)

#     return {
#         "intent": intent,
#         "selector": selector,
#         # "stepName": step_name,
#         "botMessage": reply
#     }
