from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI
from dotenv import load_dotenv
from collections import OrderedDict
import azure.cognitiveservices.speech as speechsdk
from pydantic import BaseModel
import ast
import os
import json
import re

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

model_name = "gpt-35-turbo"
deployment_name = "gpt-35-turbo"
api_version = "2024-12-01-preview"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

identify_intent_prompt = '''
You are a helpful banking assistant for tasks like e-Transfer money.
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
You are helping the user transfer money. Your job is to guide the user to click the "e-Transfer" tab on the top right of the website. The button is highlighted in yellow and labeled "e-Transfer". If the user asks questions about the button (location, color, label, or other details), you should answer clearly. Do not exceed 80 characters or 1 sentence in your reply.
'''


CLICK_ETRANSFER_BTN_PROMPT_FRANK = '''
You are helping the user transfer money. Your job is to click the "e-Transfer" tab on the top right of the website for the user. The button is highlighted in yellow and labeled "e-Transfer". If the user asks questions about the button (location, color, label, or other details), you should answer clearly. Do not exceed 80 characters or 1 sentence in your reply.
'''

check_transferee_prompt = '''
You are helping the user transfer money.
You are currently on the page of selecting a recipient. Your goal is to guide the user through selecting the intended recipient.
First, ask the user whether the person they want to transfer to is already listed on this page. Based on the user's response:
If the user indicates **yes**, ask them to click on the recipient.
If the user indicates **no**, ask them to click the 'Add New Contact' button to add the recipient.
Do not exceed 120 charaters or 2 sentences in your reply.
'''

check_transferee_prompt_frank = '''
You’re helping the user e-transfer money.

Your task happens in 3 steps:

1. Ask who they want to send money to (if not already mentioned).

2. If they mention a name, confirm if it's Bob Chen. DO NOT act yet.
   - For example: “Do you mean Bob Chen?” or “You want to send to Sophia Smith, right?”
   - Do NOT say anything like “selecting” or include any “Recipient:” line yet.
   - Just confirm the name clearly and conversationally.

3. If the user says yes or otherwise confirms, proceed casually:
   - For example: “Perfect, I'm selecting Bob Chen for you.”
   - Then add:  
     Recipient: Bob Chen

Rules:
- Only include the "Recipient:" line after user confirmation.
- Keep replies short, casual, and non-repetitive.
'''


ENTER_AMOUNT_PROMPT = '''

The user is on the page to etransfer money to a recipient. 
Your goal is to ask the user which account they want to choose as “From Account” and choose the accounts they'd like to transfer money from. Then, enter the amount they want to send. Once they've done that, click on “Continue”. They can also click on "Cancel" at any time. Keep your reply short and easy to understand. Do not exceed 2 sentences.
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

send_to_alex_choose_account_prompt = '''
You're helping the user select the source account to send money from.

1. Ask which account they want to use. DO NOT act yet.
2. If the user mentions one, confirm casually:  
   “Got it — you want to use your chequing account?”
3. Once they confirm, reply:
   “Okay, I'm selecting chequing account for you.”  
   Then include:
   VALUE: chequing (use the actual account they said)

Rules:
- Valid account values are: chequing or savings.
- Never guess. If unsure, ask again.
- Keep replies short and conversational.
'''

send_to_alex_enter_amount_prompt = '''
You're helping the user fill in amount to send money.

1. Ask how much they want to use. DO NOT act yet.
2. If the user mentions a number, confirm casually:  
   “Got it — you want to send 100 dollars?”
3. Once they confirm, reply:
   “Okay, I'm entering 100 dollars for you.”  
   Then include:
   VALUE: [a number] (use the actual number they said)

Rules:
- Valid amount are numbers.
- Never guess. If unsure, ask again.
- Keep replies short and conversational.
'''

send_to_alex_click_continue_prompt = '''
You're helping the user fill in amount to send money.

1. Ask them to double check information. For example:
   "Let's quickly double-check all the information before proceeding. Are you sending <$100> from your <savings> account to <Bob Chen>?" (Replacing things in [] with actual values)
2. Once they confirm, reply:
   “Okay, I'm clicking 'Continue' for you.”
   Then include:
   VALUE: None

Rules:
- Never guess. If unsure, ask again.
- Keep replies short and conversational.
'''


SEND_MONEY_PROMPT_FRANK = '''
You are a helpful banking assistant.

Page: {currentPage}  
Goal: {intent}

Form:
- account: {account}
- amount: {amount}

Instructions:
1. Your goal is to fill in the form with non-null values by asking the user questions. For exampple "Which account do you want to use? And how much do you want to send?"
2. You must fill in both "account" and "amount" fields.
3. If the user gives a new value, confirm it casually. Change only that field.
6. After your message, append the updated state in the format below.

Respond like this:

<your reply>

---
STATE:
{{
  "account": "chequing" or "savings" or null,
  "amount": 150 or null,
}}

Notes:
- "account" means the user's **source** account (sending from). Valid: "chequing" or "savings"
- Use lowercase for account and number for amount.
- Never include anything after the STATE block.
'''

CONFIRMATION_PROMPT = '''
You are a helpful assistant.

The user has filled out a form with the following information:
{formatted_fields}

Your job:
1. Ask the user to confirm if this information is correct. Do not present the form directly. Instead, summarize the key details in a casual way, like: "Can you confirm if you like to send $1000 from your chequing account?"
2. If the user agrees, respond casually and set "confirmed" to true.
3. If the user wants to change anything, just say so and set "confirmed" to false.
4. Do not attempt to update individual fields.
5. Keep replies short and natural.

After your reply, always append this:

---
STATE:
{{
  "confirmed": true or false
}}
'''




e_transfer = OrderedDict({
    "index.html": {
        "immediate_reply": "Click the 'e-Transfer' button on the top right of the page",
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
    },
    "success.html": {
        "immediate_reply": "Anything else I can help you with?",
        "selector": "",
        "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
        "desc": ""
    }
})

e_transfer_teller = OrderedDict({
    "index.html": {
        "immediate_reply": "I'm clicking the 'e-Transfer' button for you and you will land on the e-transfer page shortly.",
        "prompt": CLICK_ETRANSFER_BTN_PROMPT,
        "desc": "Clicked the 'E-transfer' tab",
        "action": [{"action": "click", "selector": "#nav-transfer"}],  # Frank will click the button for the user
    },
    "etransfer.html": {
        "prompt": check_transferee_prompt_frank,
        "desc": "Selected the recipient",
        "dynamic_handler": "recipient_selection",
        "action": [{"action": "click", "selector": "#contact-bob"}]  # Frank will click the recipient for the user
    },
    "send_to_alex.html": {
        "immediate_reply": "Which account do you want to transfer from? And how much?",
        "prompt": SEND_MONEY_PROMPT_FRANK,  # Optional: keep general one
        "dynamic_handler": "collect_then_act",
        "state": {"account": None, "amount": None, "confirmed": None},
        "desc": "Filled in information and clicked 'Continue'",
    },
    "confirm_transfer.html": {
        "immediate_reply": "Please double-check the information and click 'Confirm' to complete the transfer, or 'Cancel' if you want to stop.",
        "prompt": confirm_transfer_prompt,
        "desc": "Clicked 'Confirm'",
        "action": [{"action": "click", "selector": "#confirm-button, #cancel-button"}]  # Frank will click the button for the user
    },
    "success.html": {
        "immediate_reply": "Anything else I can help you with?",
        "selector": "",
        "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
        "desc": ""
    }
})

flows = {
    "e_transfer": {"grace": e_transfer, "frank": e_transfer_teller},
}


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
    # print("API cleaned_messages", cleaned_messages)
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "system", "content": (prompt)}] + cleaned_messages
    )
    print("API response:", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

def extract_bot_message_and_state(text: str) -> tuple:
    botMessage = ""
    state = {}

    parts = text.strip().split('---')
    if len(parts) >= 2:
        botMessage = parts[0].strip()

        # Find and sanitize the STATE block
        state_match = re.search(r'STATE:\s*({.*})', parts[1], re.DOTALL)
        if state_match:
            state_str = state_match.group(1)

            # Remove trailing commas before closing brace
            state_str = re.sub(r',\s*}', '}', state_str)

            try:
                state = json.loads(state_str)
            except json.JSONDecodeError as e:
                print("⚠️ JSON decode error:", e)
                state = {}
    else:
        botMessage = text.strip()

    return botMessage, state



def merge_state(existing: dict, update: dict) -> dict:
    """
    Merges a partial update into the existing state, overriding only non-null values.
    """
    for key, value in update.items():
        if value is not None:
            existing[key] = value
    return existing


def build_state_prompt(current_state, currentPage, intent, prompt=SEND_MONEY_PROMPT_FRANK):
    def safe(val):
        return json.dumps(val) if val is not None else "null"
    
    return prompt.format(
        currentPage=currentPage,
        intent=intent,
        account=safe(current_state.get("account")),
        amount=safe(current_state.get("amount")),
        confirmed=safe(current_state.get("confirmed"))
    )

def generate_actions_from_state(state: dict) -> list:
    """
    Given the current state, returns a list of UI actions to perform.
    - 'account' → action: select, selector: #from-account
    - 'amount'  → action: fill,   selector: #amount
    """
    actions = []

    if state.get("account"):
        actions.append({
            "action": "select",
            "value": state["account"],
            "selector": "#from-account"
        })

    if state.get("amount"):
        actions.append({
            "action": "fill",
            "value": state["amount"],
            "selector": "#amount"
        })

    return actions


def run_conversational_agent(messages, current_state, currentPage, intent, prompt):
    prompt = build_state_prompt(current_state, currentPage, intent, prompt=prompt)
    gpt_output = api_call(prompt=prompt, messages=messages)

    # Extract state from the GPT response
    botMessage, new_state = extract_bot_message_and_state(gpt_output)
    print("state from GPT:", new_state)
    return {
        "botMessage": botMessage,
        "state": new_state
    }

def format_fields_for_prompt(state: dict) -> str:
    return '\n'.join(f"- {k}: {v}" for k, v in state.items())

def run_confirmation_agent(messages, state, intent, current_page, conversational_prompt):
    formatted_fields = format_fields_for_prompt(state)
    prompt = CONFIRMATION_PROMPT.format(formatted_fields=formatted_fields)

    gpt_output = api_call(prompt=prompt, messages=messages)
    botMessage, extracted = extract_bot_message_and_state(gpt_output)
    print("botMessage:", botMessage)
    print("extracted:", extracted)

    confirmed = extracted.get("confirmed", False)
    if confirmed:
        actions = generate_actions_from_state(state)
        return {
            "botMessage": botMessage,
            "action": actions,
        }
    else:
        # User wants to change something → hand off to conversational agent
        conversational_response = run_conversational_agent(
            messages=messages,
            current_state=state,
            currentPage=current_page,
            intent=intent,
            prompt=conversational_prompt,
        )
        return {
            "botMessage": conversational_response["botMessage"]
        }



@app.post("/speak")
async def speak_text(request: Request):
    data = await request.json()
    text = data.get("text", "")

    # Configure Azure TTS
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("AZURE_SPEECH_KEY"), region="westus" )
    speech_config.speech_synthesis_voice_name = "en-US-FableTurboMultilingualNeural"
    speech_config.speech_synthesis_language = "en-CA"  # Force Canadian English accent

    
    # Output to audio stream
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize speech
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()   # The speech happens entirely inside this line. It triggers Azure TTS → it sends audio → plays on speaker.
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
        return { "status": "success", "text": text }
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and endpoint values?")
        return { "status": "error", "reason": str(cancellation_details.reason) }

# Grace
@app.post("/tutorbot")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    new_page_loaded = body.get("newPageLoaded", False)
    subtask_updated = body.get("substepUpdated", False)
    intent = body.get("intent") or None
    substep_flags = body.get("substep_flags", {})   # example: {"account_chosen": True}
    current_page = body.get("currentPage")    # e.g., check_transferee
    assistant = body.get("assistant", "grace")  # e.g., "grace" or "frank"
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
            if current_page in flows[intent][assistant]:
                current_step = flows[intent][assistant][current_page]
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
    # When an intent is identified, there are two situations:
    # Intent is identified && a new page just loaded, so no instruction has been sent, then send an initial instruction directly
    # Intent is identified && the user has updated a subtask but didn't send messages to the chatbot, then send the next substep instruction
    else:
        if new_page_loaded or subtask_updated:
            if intent in flows and current_page in flows[intent][assistant]:
                current_step = flows[intent][assistant][current_page]
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
            # Intent is identified && at least one instruction has been sent (user asks other questions after the next action has been highlighted), then answer their questions
            # Also need to track if the user change their intent or not here
            if intent in flows and current_page in flows[intent][assistant]:
                current_step = flows[intent][assistant][current_page]
                print("====current_step", current_step)
                res = api_call(current_step["prompt"], messages)
                return {
                    "botMessage": res
                }
    
# Frank
@app.post("/tellerbot")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    new_page_loaded = body.get("newPageLoaded", False)
    subtask_updated = body.get("substepUpdated", False)
    intent = body.get("intent") or None
    substep_flags = body.get("substep_flags", {})   # example: {"account_chosen": True}
    current_page = body.get("currentPage")    # e.g., check_transferee
    state = body.get("state", {})  # e.g., {"account": "chequing", "amount": 100, "confirmed": None}
    assistant = body.get("assistant", "grace")  # e.g., "grace" or "frank"
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
            if current_page in flows[intent][assistant]:
                current_step = flows[intent][assistant][current_page]
                if "action" in current_step:
                    print("====Current step found in flows:", current_step)
                    return {
                        "intent": intent,
                        "botMessage": current_step["immediate_reply"],
                        "action": current_step.get("action", "")  # e.g., "click" or "type"
                    }
                else:
                    print("====WIP: No action found for current step in intent:", current_step)
            else:
                print("====current_page not in flows[res]")
        else:
            print("====WIP: Intent not found in flows (GPT hallucination or not built yet)", intent)
    
    # When an intent is identified, there are two situations:
    # Intent is identified && a new page just loaded, so no instruction has been sent, then send an initial instruction directly
    # Intent is identified && the user has updated a subtask but didn't send messages to the chatbot, then send the next substep instruction
    else:
        if intent in flows and current_page in flows[intent][assistant]:
            current_step = flows[intent][assistant][current_page]
            # Step 3: Handle dynamic behavior before static substeps
            handler_type = current_step.get("dynamic_handler")
            print("====Handler type:", handler_type)
            if handler_type == "recipient_selection":
                clarification = api_call(current_step["prompt"], messages)
                print("====Raw GPT response:", clarification)

                # Extract recipient from a line like "Recipient: Bob Chen"
                recipient_name = ""
                for line in clarification.splitlines():
                    if line.lower().startswith("recipient:"):
                        recipient_name = line.split(":", 1)[1].strip()
                        break
                print("====Extracted recipient name:", recipient_name)
                selector = "#contact-bob" if "bob" in recipient_name.lower() else ""
                print("====Recipient selector:", selector)
                if selector:
                    # Extract the first sentence (before newline or period)
                    first_sentence = clarification.splitlines()[0].strip()
                    return {
                        "intent": intent,
                        "botMessage": first_sentence,
                        # "selector": selector,
                        "action": current_step.get("action", "")
                    }
                else:
                    # Couldn't match recipient – fallback to full response
                    return {
                        "intent": intent,
                        "botMessage": clarification,
                        "selector": "",
                        "action": ""
                    }
            
            elif handler_type == "collect_then_act":
                print("====Current state:", state)
                # Step 1: Initial guidance message if state is empty
                if not state:
                    return {
                        "botMessage": current_step["immediate_reply"],
                        "state": current_step["state"],
                    }

                print("step 2")
                # Step 2: If state is partially filled, continue data collection
                if not (state.get("account") and state.get("amount")):
                    gpt_response = run_conversational_agent(
                        messages=messages,
                        current_state=state,
                        currentPage=current_page,
                        intent=intent,
                        prompt=current_step.get("prompt")
                    )

                    # ✅ After filling both fields, return actions too
                    filled = gpt_response.get("state", {})
                    if filled.get("account") and filled.get("amount"):
                        gpt_response["action"] = generate_actions_from_state(filled)

                    return gpt_response

                print("step 3")
                # Step 3: If both fields are filled, confirm with user
                confirmation_result = run_confirmation_agent(messages, state, current_page, intent, conversational_prompt=current_step.get("prompt"))

                # Either confirmed → return actions, or not → conversational agent will take over
                return confirmation_result
            
            # If no substeps, proceed with the current step
            if "action" in current_step:
                return {
                    "intent": intent,
                    "botMessage": current_step["immediate_reply"],
                    "action": current_step.get("action", "")
                }
            else:
                print("====WIP: No action found for current step in intent:", current_step)
        else:
            print("====WIP: Current page not found in flows for intent:", intent, current_page)


### Another endpoint to add payees

payees = {}  # key: user_id or session_id, value: list of payees

class Payee(BaseModel):
    name: str
    account: str
    user_id: str  # or session_id

@app.post("/api/add_payee")
async def add_payee(payee: Payee):
    user_payees = payees.setdefault(payee.user_id, [])
    user_payees.append({
        "name": payee.name,
        "account": payee.account
    })
    return {"status": "success", "payees": user_payees}