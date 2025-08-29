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


# CLICK_ETRANSFER_BTN_PROMPT = '''
# You are helping the user transfer money. Your job is to guide the user to click the "e-Transfer" tab on the top of the website. The button is highlighted in yellow and labeled "e-Transfer". If the user asks questions about the button (location, color, label, or other details), you should answer clearly. Do not exceed 80 characters or 1 sentence in your reply.
# '''

# CLICK_ETRANSFER_BTN_PROMPT_FRANK = '''
# You are helping the user transfer money. Your job is to click the "e-Transfer" tab on the top of the website for the user. The button is highlighted in yellow and labeled "e-Transfer". If the user asks questions about the button (location, color, label, or other details), you should answer clearly. Do not exceed 80 characters or 1 sentence in your reply.
# '''

# CHECK_TRANSFEREE_PROMPT = '''
# You are helping the user transfer money.
# You are currently on the page of selecting a recipient. Your goal is to guide the user through selecting the intended recipient.
# First, ask the user whether the person they want to transfer to is already listed on this page. Based on the user's response:
# If the user indicates **yes**, ask them to click on the recipient.
# If the user indicates **no**, ask them to click the 'Add New Contact' button to add the recipient.
# Do not exceed 120 charaters or 2 sentences in your reply.
# '''

# CONFIRM_TRANSFER_PROMPT = '''
# The user is on the last step to e-transfer money. Your goal is to guide the user to double check the information and click on 'Confirm' if want to continue, otherwise click 'Cancel' to cancel the transaction.  Keep your reply short and easy to understand. Do not exceed 2 sentences.
# '''

INTENT_PROMPT = '''
You are a helpful banking assistant. Your job is to identify the user's goal. Response a goal from e_transfer (to send people money), check_activity (check account activity/balance or download statement) or pay_bill (pay bill to some company or organization).
If the user's goal is clear, reply with exactly one of the intent names above.
If the user's goal is unclear, ambiguous, or missing, respond with exactly: clarification_required.
Do not guess. Do not explain your choice. Do not include any punctuation or extra words.
'''

INTENT_CLARIFICATION_PROMPT = """
The user’s intent is unclear. Your job is to ask a short, polite follow-up question that will help determine whether the user wants to:
- e_transfer
- pay_bill
- check_balance

You must respond with **a single, clear question**, no longer than 20 words. Do not guess the user's intent. Only ask one question at a time.

Examples:
- Would you like to transfer money, pay a bill, or check your balance?
- What banking task would you like to complete today?
"""

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
4. After your message, append the updated state in the format below.

Respond like this:

<your reply>

STATE:
{{
  "account": "chequing" or "savings" or null,
  "amount": 150 or null,
}}

Notes:
- "account" means the user's **source** account (sending from). Valid values are only: "chequing" or "savings"
- Use lowercase for account and number for amount.
- Never include anything after the STATE block.
'''

CONFIRMATION_PROMPT = '''
You are a helpful assistant.

The user has filled out a form with the following information:
{formatted_fields}

Your job:
1. Ask them to confirm casually — e.g., " I have filled in the information. $10 is the amount that will be sent from your chequing account. Would you like to continue?"
2. Once the user agrees, respond casually and set "confirmed" to true. "confirmed" is originally set to false.
3. If the user wants to change anything, say so and set "confirmed" to false.
4. Do not attempt to update individual fields.
5. Keep replies short and natural.

Your response **must** include:
- A conversational message first
- Then a `STATE` block on a new line

Example format:

I have filled in the information. $10 is the amount that will be sent from your chequing account. Would you like to continue? 

STATE:
{{
  "confirmed": false
}}
'''

YESNO_CLASSIFIER_PROMPT = """
Does the user say YES to a yes/no question?

Respond with exactly one word:
- yes
- no
- unclear

Do NOT explain or add punctuation. Do NOT include any extra words.
"""

# Stage 1: Classify or ask for clarification
CLASSIFICATION_DECISION_PROMPT = (
    "Based on the conversation so far, classify the user's intent into one of the following options: "
    "'{label_list}'.\n\n"
    "If the user's goal is clear, reply with exactly one of the option names.\n"
    "If the user's goal is unclear, ambiguous, or missing, respond with exactly: clarification_required."
    "Do not add any punctuation or extra words other than one of the provided options or 'clarification_required'."
)
# Stage 2: Generate a clarification question
CLARIFICATION_PROMPT = (
    "The user’s intent is unclear. Your job is to ask a short, polite follow-up question "
    "that will help determine whether the user wants to: {label_list}."
)

# --- Back-intent classifier prompt (binary) ---
GO_BACK_PROMPT = """
You are a binary classifier. Decide if the user is asking to go back to the previous screen in the app.
Output exactly one token: go_back or none.

Consider "go back", "back", "previous", "previous screen/page", "take me back",
"back to [X]" (e.g., back to accounts), "undo last step", "return to the last page".
Treat "undo last step" or "change selection" as go_back if it implies returning to the prior screen.

Do NOT trigger for unrelated uses of "back" (e.g., "back pain", "background", "cashback", "back soon").

Conversation (latest message last):
{messages}
Answer:
"""


def wants_navigation_back(messages) -> bool:
    """
    Uses GPT to decide if the user wants to navigate back.
    Returns True if GPT says 'go_back'; otherwise False.
    """
    try:
        label = (api_call(GO_BACK_PROMPT, messages) or "").strip().lower()
        return label.startswith("go_back")
    except Exception:
        return False  # fail safe: if classifier fails, don't navigate


# Grace - Alex
e_transfer_tutor = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_etransfer": {
                "immediate_reply": "Click the 'e-Transfer' button on the top of the page",
                # "prompt": CLICK_ETRANSFER_BTN_PROMPT,
                "action": [{"action": "highlight", "selector": "#nav-transfer"}],  # Grace will highlight the button for the user
                "desc": "Clicked the 'E-transfer' tab"
            }
        })
    },
    "etransfer.html": {
        "substeps": OrderedDict({
            "select_recipient": {
                "immediate_reply": "Please select the recipient you want to transfer money to.",
                "action": [],
                # "prompt": CHECK_TRANSFEREE_PROMPT,
                "desc": "Selected the recipient"
            }
        })
    },
    "send_to_alex.html": {
        "substeps": OrderedDict({
            "choose_account": {
                "immediate_reply": "Please choose the account you want to transfer from.",
                "action": [{"action": "highlight", "selector": "#from-account"}],  # Grace will highlight the account selector for the user
                "completion_condition": "account_chosen",  # flag name
                "desc": "Selected the account to transfer from"
            },
            "enter_amount": {
                "immediate_reply": "Now enter the amount.",
                "action": [{"action": "highlight", "selector": "#amount"}],  # Grace will highlight the amount input for the user
                "completion_condition": "amount_entered",
                "desc": "Entered the amount to transfer"
            },
            "continue": {
                "immediate_reply": "Now click 'Continue'.",
                "action": [{"action": "highlight", "selector": "#send-button"}],  # Grace will highlight the continue button for the user
                "completion_condition": "continue_clicked",
                "desc": "Clicked 'Continue'"
            },
        }),
    },
    "confirm_transfer.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Please double-check the information and click 'Confirm' to complete the transfer, or 'Cancel' if you want to stop.",
                "selector": "#confirm-button, #cancel-button",
                "action": [{"action": "highlight", "selector": "#confirm-button, #cancel-button"}],  # Grace will highlight the confirm and cancel buttons for the user
                # "prompt": CONFIRM_TRANSFER_PROMPT,
                "desc": "Clicked 'Confirm'"
            }
        })
    },
    "success.html": {
        "substeps": OrderedDict({
            "success_message": {
                "immediate_reply": "Anything else I can help you with?",
                "selector": "",
                "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
                "desc": ""
            }
        })
    }
})

e_transfer_teller = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_etransfer": {
                "immediate_reply": "I'm clicking the 'e-Transfer' button for you and you will land on the e-transfer page shortly.",
                # "prompt": CLICK_ETRANSFER_BTN_PROMPT,
                "desc": "Clicked the 'E-transfer' tab",
                "action": [{"action": "click", "selector": "#nav-transfer"}],  # Frank will click the button for the user
            }
        })
    },
    "etransfer.html": {
        "substeps": OrderedDict({
            "select_recipient": {
                "desc": "Selected the recipient",
                "immediate_reply": "Who would you like to send money to?",
                "dynamic_handler": "classification_handler",
                "prompt": "The user is on the page of selecting a recipient of a potential eTransfer. There are 3 users on the page. Your goal is to guide the user through selecting the intended recipient.",
                "options": {
                    "Bob Chen": {
                        "action": [{"action": "highlight", "selector": "#contact-bob", "immediate_reply": "Can you confirm that you're sending money to Bob Chen?"}]
                    },
                    "Sophia Smith": {},
                    "David Kim": {}
                },
                "completion_condition": "select_recipient",  # flag name
            },
            "confirm_recipient": {
                "dynamic_handler": "confirmation_handler",
                "action": [{"action": "click", "selector": "#contact-bob", "immediate_reply": "Thank you for confirming. I'm selecting Bob Chen for you."}],  # You can generate this dynamically too
                "completion_condition": "confirm_recipient"
            },
        })
    },
    "send_to_alex.html": {
        "substeps": OrderedDict({
            "choose_account": {
                "immediate_reply": "Which account do you want to send money from?",
                "dynamic_handler": "selection_handler",
                "action": [{"action": "highlight", "selector": "#from-account"}],
                "options": {
                    "chequing account": {
                        "action": [{"action": "select", "selector": "#from-account", "value": "chequing", "immediate_reply": "I'm selecting your Chequing account for the payment."}],
                    },
                    "savings account": {
                        "action": [{"action": "select", "selector": "#from-account", "value": "savings", "immediate_reply": "I'm selecting your Savings account for the payment."}],
                    },
                },
                "desc": "Filled in 'from account'",
                "completion_condition": "account_chosen",  # flag name
            },
            "enter_amount": {
                "immediate_reply": "How much do you want to send?",
                "dynamic_handler": "fill_handler",
                "field": "Amount ($):",
                "value": "Numbers only, it could be a whole number or a decimal.", 
                "action": [{"action": "fill", "selector": "#amount", "immediate_reply": "I'm filling in the amount for you."}],
                "desc": "Filled in amount",
                "completion_condition": "amount_entered",  # flag name
            },
            "confirm": {
                "immediate_reply": "Okay do you want to continue with the payment or cancel it?",
                "dynamic_handler": "selection_handler",
                "options": {
                    "cancel": {
                        "action": [{"action": "click", "selector": "#cancel-button", "immediate_reply": "I'm clicking 'Cancel' for you."}],
                    },
                    "continue": {
                        "action": [{"action": "click", "selector": "#send-button", "immediate_reply": "I'm clicking 'Continue' for you."}],
                    },
                },
            }
        }),  
    },
    "confirm_transfer.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Because this is the final step, you need to take the action yourself. Please double-check the information and click 'Confirm' to complete the transfer, or 'Cancel' if you want to stop.",
                # "prompt": CONFIRM_TRANSFER_PROMPT,
                "desc": "Clicked 'Confirm'",
                "action": [{"action": "highlight", "selector": "#confirm-button, #cancel-button"}]  # Frank will click the button for the user
            }
        })
    },
    "success.html": {
        "substeps": OrderedDict({
            "success": {
                "immediate_reply": "Anything else I can help you with?",
                "selector": "",
                "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
                "desc": ""
            }
        })
    }
})

# Grace - Alex
pay_bill_tutor = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_etransfer": {
                "immediate_reply": "Click the 'Pay Bills' button on the top of the page",
                # "prompt": CLICK_ETRANSFER_BTN_PROMPT,
                "action": [{"action": "highlight", "selector": "#nav-paybill"}],  # Grace will highlight the button for the user
                "desc": "Clicked the 'Pay Bill' tab"
            }
        })
    },
    "pay_bill.html": {
        "substeps": OrderedDict({
            "select_recipient": {
                "immediate_reply": "Please select one of the saved payees or add a new payee.",
                "action": [],
                # "prompt": CHECK_TRANSFEREE_PROMPT,
                "desc": "Selected the recipient"
            }
        })
    },
    "payee.html": {
        "substeps": OrderedDict({
            "choose_account": {
                "immediate_reply": "Please choose the account you want to transfer from.",
                "action": [{"action": "highlight", "selector": "#from-account"}],  # Grace will highlight the account selector for the user
                "completion_condition": "account_chosen",  # flag name
                "desc": "Selected the account to transfer from"
            },
            "enter_amount": {
                "immediate_reply": "Now enter the amount.",
                "action": [{"action": "highlight", "selector": "#amount"}],  # Grace will highlight the amount input for the user
                "completion_condition": "amount_entered",
                "desc": "Entered the amount to transfer"
            },
            "continue": {
                "immediate_reply": "Please click 'Continue' if the account and the amount are correct.",
                "action": [{"action": "highlight", "selector": "#auto-pay, #send-button"}],  # Grace will highlight the continue button for the user
                "completion_condition": "continue_clicked",
                "desc": "Clicked 'Continue'"
            },
        }),
    },
    "confirm_bill.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Please double-check the information and click 'Confirm' to complete the transfer, or 'Cancel' if you want to stop.",
                "action": [{"action": "highlight", "selector": "#confirm-button, #cancel-button"}],  # Grace will highlight the confirm and cancel buttons for the user
                # "prompt": CONFIRM_TRANSFER_PROMPT,
                "desc": "Clicked 'Confirm'"
            }
        })
    },
    "success.html": {
        "substeps": OrderedDict({
            "success_message": {
                "immediate_reply": "Anything else I can help you with?",
                "selector": "",
                "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
                "desc": ""
            }
        })
    },
    "add_payee.html": {
        "substeps": OrderedDict({
            "fill_name": {
                "immediate_reply": "Please enter Payee's name.",
                "action": [{"action": "highlight", "selector": "#payee-name"}],  # Grace will highlight the account selector for the user
                "completion_condition": "name_filled",  # flag name
            },
            "fill_account": {
                "immediate_reply": "Now enter the 11-digits-long account number.",
                "action": [{"action": "highlight", "selector": "#account-number"}],  # Grace will highlight the amount input for the user
                "completion_condition": "account_filled",
            },
            "continue": {
                "immediate_reply": "Please click 'Continue' if the name and the account number are correct.",
                "action": [{"action": "highlight", "selector": "#add-payee"}],  # Grace will highlight the continue button for the user
                "completion_condition": "continue_clicked",
            },
        })
    },
    "confirm_payee.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Please double-check the information and click 'Confirm' to add this payee, or 'Cancel' if you want to stop.",
                "action": [{"action": "highlight", "selector": "#confirm-button, #cancel-button"}],  # Grace will highlight the confirm and cancel buttons for the user
                # "prompt": CONFIRM_TRANSFER_PROMPT,
            }
        })
    },
    "payee_added.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Please click 'Back to Pay Bills' to return to the Pay Bills page.",
                "action": [{"action": "highlight", "selector": "#back-pay-bill"}],  # Grace will highlight the confirm and cancel buttons for the user
                # "prompt": CONFIRM_TRANSFER_PROMPT,
            }
        })
    }
})

pay_bill_teller = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_etransfer": {
                "immediate_reply": "I'm clicking the 'Pay Bills' button for you and you will land on the page shortly.",
                # "prompt": CLICK_ETRANSFER_BTN_PROMPT,
                "desc": "Clicked the 'Pay Bills' tab",
                "action": [{"action": "click", "selector": "#nav-paybill"}],  # Frank will click the button for the user
            }
        })
    },
    "pay_bill.html": {
        "substeps": OrderedDict({
            "select_recipient": {
                "desc": "Selected the recipient",
                "immediate_reply": "Who would you like to pay your bill to? I can add a new payee for you too.",
                "dynamic_handler": "classification_handler",
                "prompt": "The user is on the page of selecting a recipient of a potential bill payment. There are 3 recipients on the page. Your goal is to guide the user through selecting the intended recipient.",
                "options": {
                    "Bell": {
                        "action": [{"action": "highlight", "selector": "#bell", "immediate_reply": "Can you confirm that you're paying your bill to Bell?"}]
                    },
                    "BC Hydro": {},
                    "Telus Mobile": {},
                    "Add New Payee": {
                        "action": [{"action": "highlight", "selector": "#add-contact", "immediate_reply": "Can you confirm that you're adding a new payee?"}]
                    }
                },
                "completion_condition": "select_recipient",  # flag name
            },
            "confirm": {
                "dynamic_handler": "confirmation_handler",
                "action": [{"action": "click", "immediate_reply": "Thank you for confirming. I'm clicking it for you shortly."}],  # You can generate this dynamically too
                "completion_condition": "confirm_recipient"
            }
        })
    },
    "payee.html": {
        "substeps": OrderedDict({
            "choose_account": {
                "immediate_reply": "Which account do you want to pay from?",
                "dynamic_handler": "selection_handler",
                "action": [{"action": "highlight", "selector": "#from-account"}],
                "options": {
                    "chequing account": {
                        "action": [{"action": "select", "selector": "#from-account", "value": "chequing", "immediate_reply": "I'm selecting your Chequing account for the payment."}],
                    },
                    "savings account": {
                        "action": [{"action": "select", "selector": "#from-account", "value": "savings", "immediate_reply": "I'm selecting your Savings account for the payment."}],
                    },
                },
                "desc": "Filled in 'from account'",
                "completion_condition": "account_chosen",  # flag name
            },
            "enter_amount": {
                "immediate_reply": "How much do you want to pay?",
                "dynamic_handler": "fill_handler",
                "field": "Amount ($):",
                "value": "Numbers only, it could be a whole number or a decimal.", 
                "action": [{"action": "fill", "selector": "#amount", "immediate_reply": "I'm filling in the amount for you."}],
                "completion_condition": "amount_entered",  # flag name
            },
            "confirm": {
                "immediate_reply": "Okay do you want to continue with the payment or cancel it?",
                "dynamic_handler": "selection_handler",
                "options": {
                    "cancel": {
                        "action": [{"action": "click", "selector": "#cancel-button", "immediate_reply": "I'm clicking 'Cancel' for you."}],
                    },
                    "continue": {
                        "action": [{"action": "click", "selector": "#send-button", "immediate_reply": "I'm clicking 'Continue' for you."}],
                    },
                },
            },
        }),  
    },
    "confirm_bill.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Do you want to confirm this payment or cancel it?",
                "dynamic_handler": "confirmation_handler",
                "action_description": "a bill payment",
                "action": [{"action": "highlight", "selector": "#confirm-button", "immediate_reply": "Since this action cannot be reversed, I need you to confirm twice. Do you want to confirm this bill payment? Yes or No."}],  # Frank will click the button for the user
                "completion_condition": "confirm_transfer",  # flag name
            },
            "double_confirm_transfer": {
                "immediate_reply": "",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "click", "selector": "#confirm-button", "immediate_reply": "Thank you for confirming. I'm clicking 'Confirm' for you."}]  # Frank will click the button for the user
                    },
                    "no": {
                        "action": [{"action": "click", "selector": "#cancel-button", "immediate_reply": "No problem. I'm clicking 'Cancel' for you."}]  # Frank will
                    }
                }
            }
        })
    },
    "success.html": {
        "substeps": OrderedDict({
            "success": {
                "immediate_reply": "Anything else I can help you with?",
                "selector": "",
                "prompt": "The user has successfully completed the transfer. Ask if they have more questions or click 'Home' to return to the homepage. Do not exceed 50 characters.",
                "desc": ""
            }
        })
    },
    "add_payee.html": {
        "substeps": OrderedDict({
            "fill_name": {
                "immediate_reply": "What's your Payee's name? Can you spell it for me?",
                "dynamic_handler": "fill_handler",
                "field": "Payee's name",
                "value": "It will be an organization name.", 
                "example": "e.g. Bell, BC Hydro, Telus Mobile",
                "action": [{"action": "fill", "selector": "#payee-name", "immediate_reply": "I'm filling in the name for you."}],  # Grace will highlight the account selector for the user
                "completion_condition": "name_filled",  # flag name
            },
            "fill_account": {
                "immediate_reply": "What's the Payee's account number? It should be 11-digits-long.",
                "dynamic_handler": "fill_handler",
                "field": "Account number",
                "value": "It will be numbers or space only.", 
                "action": [{"action": "fill", "selector": "#account-number", "immediate_reply": "I'm filling in the account number for you."}],  # Grace will highlight the amount input for the user
                "completion_condition": "account_filled",
            },
            "continue": {
                "immediate_reply": "Okay do you want to continue adding this payee or cancel it?",
                "dynamic_handler": "selection_handler",
                "options": {
                    "cancel": {
                        "action": [{"action": "click", "selector": "#cancel-button", "immediate_reply": "I'm clicking 'Cancel' for you."}],
                    },
                    "add payee": {
                        "action": [{"action": "click", "selector": "#add-payee", "immediate_reply": "I'm clicking 'Add a Payee' for you."}],
                    },
                },
            },
        })
    },
    "confirm_payee.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Because this is the final step, you need to take the action yourself. Click 'Confirm' to add this payee, or 'Cancel' to stop.",
                "action": [{"action": "highlight", "selector": "#confirm-button, #cancel-button"}],  # Grace will highlight the confirm and cancel buttons for the user
                # "prompt": CONFIRM_TRANSFER_PROMPT,
            }
        })
    },
    "payee_added.html": {
        "substeps": OrderedDict({
            "confirm_transfer": {
                "immediate_reply": "Do you want to go back to the Pay Bills page to pay your bill?",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "click", "selector": "#back-pay-bill", "immediate_reply": "I'm clicking 'Back to Pay Bills' for you."}]
                    },
                    "no": {
                        "action": [{"immediate_reply": "No problem. Let me know if you need anything else."}]
                    }
                },
            }
        })
    }
})

# Grace - Alex
check_activity_tutor = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_activity": {
                "immediate_reply": "Would you like to check the balance of your checking account or savings account?",
                "dynamic_handler": "classification_handler",
                "completion_condition": "account_chosen",  # flag name
                "desc": "User selects account type to view activity for",
                "options": {
                    "chequing_account": {
                        "action": [{"action": "highlight", "selector": "#view_checking_activity", "immediate_reply": "Click the 'View Activity' button highlighted in yellow."}],  # Grace will highlight the button for the user
                        "desc": "Clicked 'View Activity' for Chequing"
                    },
                    "savings_account": {
                        "action": [{"action": "highlight", "selector": "#view_saving_activity", "immediate_reply": "Click the 'View Activity' button highlighted in yellow."}],  # Grace will highlight the button for the user
                        "desc": "Clicked 'View Activity' for Savings"
                    }
                }
            }
        })
    },
    "chequing_activity.html": {
        "substeps": OrderedDict({
            "download_chequing_statement": {
                "completion_condition": "download_chequing_statement",
                "immediate_reply": "Here's your chequing account activity. Would you like to download the statement?",
                # "prompt": "Ask the user if they'd like to download the chequing statement. If they say yes, tell them that you're clicking the download button for them.",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "highlight", "selector": "#chequing-statement-download", "immediate_reply": "Click 'Download Statement' highlighted in yellow."}]
                    },
                    "no": {
                        "action": [{"action": "", "selector": "", "immediate_reply": "No problem. Let me know if you need anything else."}]
                    }
                },
                "desc": "Prompted user to download chequing statement"
            },
            "success": {
                "immediate_reply": "Anything else I can help you with?"
            }
        })
    },
    "savings_activity.html": {
        "substeps": OrderedDict({
            "download_saving_statement": {
                "completion_condition": "download_saving_statement",
                "immediate_reply": "Here's your savings account activity. Would you like to download the statement?",
                # "prompt": "Ask the user if they'd like to download the chequing statement. If they say yes, tell them that you're clicking the download button for them.",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "highlight", "selector": "#saving-statement-download", "immediate_reply": "Click 'Download Statement' highlighted in yellow."}]
                    },
                    "no": {
                        "action": [{"action": "", "selector": "", "immediate_reply": "No problem. Let me know if you need anything else."}]
                    }
                },
                "desc": "Prompted user to download savings statement"
            },
            "success": {
                "immediate_reply": "Anything else I can help you with?"
            }
        })
    }
})
# TODO: selector -> action

check_activity_teller = OrderedDict({
    "index.html": {
        "substeps": OrderedDict({
            "click_activity": {
                "immediate_reply": "Would you like to check the balance of your checking account or savings account?",
                "dynamic_handler": "classification_handler",
                "completion_condition": "account_chosen",  # flag name
                "desc": "User selects account type to view activity for",
                "options": {
                    "chequing_account": {
                        # "selector": "#view_checking_activity",
                        "desc": "Clicked 'View Activity' for Chequing",
                        "action": [{"action": "click", "selector": "#view_checking_activity", "immediate_reply": "I'm clicking the 'View Activity' button for your chequing account"}]  # Frank will click the button for the user
                    },
                    "savings_account": {
                        # "selector": "#view_saving_activity",
                        "desc": "Clicked 'View Activity' for Savings",
                        "action": [{"action": "click", "selector": "#view_saving_activity",  "immediate_reply": "I'm clicking the 'View Activity' button for your savings account"}]  # Frank will click the button for the user
                    }
                }
            }
        })
    },
    "chequing_activity.html": {
        "substeps": OrderedDict({
            "download_chequing_statement": {
                "completion_condition": "chequing_statement_downloaded",
                "immediate_reply": "Here's your chequing account activity. Would you like to download the statement for your chequing account?",
                # "prompt": "Ask the user if they'd like to download the chequing statement. If they say yes, tell them that you're clicking the download button for them.",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "click", "selector": "#chequing-statement-download", "immediate_reply": "I'm clicking 'Download Statement' for you. Your statement will be downloaded shortly."}]
                    },
                    "no": {
                        "action": [{"action": "", "selector": "", "immediate_reply": "No problem. Let me know if you need anything else."}]
                    }
                },
                "desc": "Prompted user to download chequing statement"
            },
            "success": {
                "immediate_reply": "Anything else I can help you with?"
            }
        })
    },
    "savings_activity.html": {
        "substeps": OrderedDict({
            "download_saving_statement": {
                "completion_condition": "download_saving_statement",
                "immediate_reply": "Here's your savings account activity. Would you like to download the statement?",
                # "prompt": "Ask the user if they'd like to download the chequing statement. If they say yes, tell them that you're clicking the download button for them.",
                "dynamic_handler": "yesno_handler",
                "options": {
                    "yes": {
                        "action": [{"action": "click", "selector": "#saving-statement-download", "immediate_reply": "I'm clicking 'Download Statement' for you. Your statement will be downloaded shortly."}]
                    },
                    "no": {
                        "action": [{"action": "", "selector": "", "immediate_reply": "No problem. Let me know if you need anything else."}]
                    }
                },
                "desc": "Prompted user to download savings statement"
            },
            "success": {
                "immediate_reply": "Anything else I can help you with?"
            }
        })
    }
})

flows = {
    "e_transfer": {"grace": e_transfer_tutor, "frank": e_transfer_teller},
    "check_activity": {"grace": check_activity_tutor, "frank": check_activity_teller},
    "pay_bill": {"grace": pay_bill_tutor, "frank": pay_bill_teller},
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
    """
    Extracts the assistant's message and state from a GPT response.

    Assumes the format:
        <assistant message>
        STATE:
        {
          "account": ...,
          ...
        }
    """
    botMessage = ""
    state = {}

    # Split on 'STATE:' to separate message from state
    parts = text.strip().split("STATE:", 1)

    # Part before STATE is the message
    if parts:
        botMessage = parts[0].strip()

    # Part after STATE should contain the JSON state
    if len(parts) == 2:
        state_str = parts[1].strip()

        # Clean up any trailing commas before closing braces
        state_str = re.sub(r',\s*\n*\s*}', '}', state_str)

        try:
            state = json.loads(state_str)
        except json.JSONDecodeError as e:
            print("⚠️ JSON decode error:", e)
            state = {}

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

def run_confirmation_agent(messages, state):
    formatted_fields = format_fields_for_prompt(state)
    prompt = CONFIRMATION_PROMPT.format(formatted_fields=formatted_fields)

    gpt_output = api_call(prompt=prompt, messages=messages)
    botMessage, new_state = extract_bot_message_and_state(gpt_output)
    print("botMessage:", botMessage)
    print("extracted:", new_state)

    # ✅ Merge the confirmed value into state
    updated_state = merge_state(state, new_state)
    actions = generate_actions_from_state(updated_state)
    return {
        "botMessage": botMessage,
        "action": actions,
        "state": updated_state,
    }

# TODO: if user wants to change anything (amount, account)


# Grace - Alex
def yesno_handler(new_page_loaded, messages, substep, intent):
    # 1. Ask the yes/no question if newPageLoaded
    print("====Yes/No classification handler called")
    # print("substep", substep)
    if new_page_loaded:
        print("New page loaded, asking yes/no question...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
            # "botMessage": "test"
        }

    # 2. Classify user response
    recent_messages = messages[-2:] if len(messages) >= 2 else messages
    classification = api_call(YESNO_CLASSIFIER_PROMPT, recent_messages)
    print("Classification result:", classification)
    # print("Substep:", substep)

    if classification.lower() == "yes":
        return {
            "intent": intent,
            "action": substep["options"]["yes"]["action"],
            "substep_flags": {substep.get("completion_condition"): True}
        }
    elif classification.lower() == "no":
        return {
            "intent": intent,
            "action": substep["options"]["no"]["action"],
            "substep_flags": {substep.get("completion_condition"): True}
        }
    else:
        return {
            "intent": intent,
            "botMessage": "Sorry, could you please clarify?"
        }

# Grace - Alex
def classification_handler(substep, messages, intent, new_page_loaded=False):
    print("====Classification handler called")
    # 1. Ask the yes/no question if newPageLoaded
    # print("substep", substep)
    if new_page_loaded:
        print("New page loaded, asking yes/no question...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
        }

    options = substep.get("options", {})
    if not options:
        raise ValueError("Substep is missing 'options' for classification.")

    label_list = "', '".join(options.keys())
    print(f"Classifying what user wants with options: {label_list}")
    classification_prompt = CLASSIFICATION_DECISION_PROMPT.format(label_list=label_list)
    if substep.get("prompt"):
        classification_prompt += "\n\n" + substep["prompt"]

    result = api_call(classification_prompt, messages).strip().lower()

    if result == "clarification_required":
        clarification_prompt = CLARIFICATION_PROMPT.format(label_list=label_list)
        clarification_question = api_call(clarification_prompt, messages)

        return {
            "intent": intent,
            "action": "",
            "botMessage": clarification_question,
        }

    # If reply is matched
    for key in options:
        if result == key.lower():
            option = options[key]
            return {
                "intent": intent,
                "action": option.get("action", []),  # e.g., [{"action": "click", "selector": "#view_checking_activity"}]
                "substep_flags": {substep.get("completion_condition"): True}
            }
    print(f"No match found for classification result: {result}")
    # No match and not clarification_required
    return {
        "intent": intent,
        "action": "",
        "botMessage": "Sorry, I couldn’t understand. Can you tell me what you want to do?"
    }

# Stage 1: Classify or ask for clarification
SELECTION_PROMPT = (
    "The bot is asking the user to select one from a few options. Based on the conversation so far, classify the user's selection into one of the following options that's closer to user's selection: "
    "'{label_list}'.\n\n"
    "If the user's selection is clear, reply with exactly one of the option names.\n"
    "If the user's selection is unclear, ambiguous, or missing, respond with exactly: clarification_required."
    "Do not add any punctuation or extra words other than the option name or 'clarification_required'."
)

SELECTION_CLARIFICATION_PROMPT = (
    "The user’s selection is unclear. Your job is to ask a short, polite follow-up question "
    "that will help determine whether the user wants to choose: {label_list}."
)

def selection_handler(substep, messages, intent, new_page_loaded):
    """
    Handles selection of options from a list, e.g., account selection.
    """
    print("====Selection handler called")
    # 1. Ask the selection question if newPageLoaded
    if new_page_loaded:
        print("New page loaded, asking selection question...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
        }
    
    options = substep.get("options", {})
    if not options:
        raise ValueError("Substep is missing 'options' for selection.")
    label_list = "', '".join(options.keys())

    # 2. Classify user response
    print(f"Classifying what user wants with options: {label_list}")
    recent_messages = messages[-1:] if len(messages) >= 1 else messages
    selection = api_call(SELECTION_PROMPT.format(label_list=label_list), recent_messages)
    print("Selection result:", selection)

    if selection == "clarification_required":
        clarification_prompt = CLARIFICATION_PROMPT.format(label_list=label_list)
        clarification_question = api_call(clarification_prompt, messages)

        return {
            "intent": intent,
            "action": "",
            "botMessage": clarification_question,
        }

    if selection in options:
        print(f"User selected: {selection}")
        option = options[selection.lower()]
        return {
            "intent": intent,
            "action": option.get("action", []),
            "substep_flags": {substep.get("completion_condition"): True}
        }
    
    return {
        "intent": intent,
        "botMessage": "Sorry, I couldn’t understand your choice. Could you please tell me what you want to select?"
    }

FILL_PROMPT = '''You are a financial assistant helping a user fill in a field on a banking website.
The field is: {field}.
User's answer needs to satisfy the requirement: {value}.
Your task is to extract the relevant value from the user's message and return it. 
If the user message is clear and contains the required value, return that value.
The user message may contain the spelling of words, treat it as a word and extract the value.
If the user message is not clear, return "clarification_required" along with your reason.
You should return either a value that satisfies the requirement or "clarification_required" along with your reason.
'''

FILL_CLARIFICATION_PROMPT = (
    "The user’s number is unclear. Your job is to ask a short, polite follow-up question "
    "that will help a user fill in a field on a banking website."
    "The field is: {field}."
    "User's answer needs to satisfy the requirement: {value}"
)

PAYEE_NAME_CLEAN_PROMPT = (
    "The user is trying to fill in the name of a payee. They were asked to provide a name of an organization and then spell it out. "
    "Based on the user's message, extract the payee's name. Do not include the spelling of the name, just the name itself, so that there's no duplicate information. "
    "Do not include any other text, such as 'Payee name:', just the name of the payee. "
)

def extract_number(text: str) -> str:
    """Return the first number found in text, or '' if none."""
    if not text:
        return ""
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else ""

def fill_handler(substep, messages, intent, new_page_loaded):
    """
    Handles filling in a field, e.g., entering an amount.
    """
    print("====Fill handler called")
    # 1. Ask the fill question if newPageLoaded
    if new_page_loaded:
        print("New page loaded, asking immediate reply...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
        }

    # 2. Classify user response
    # messages is a list of dicts like {"role": "user", "content": "..."}
    recent_messages = messages[-5:] if len(messages) >= 5 else messages
    recent_messages = [
        {
            **m,
            "content": re.sub(r'(?<=\d) (?=\d)', '', m["content"])  # remove spaces between digits
        }
        for m in recent_messages
    ]
    print("User message:", recent_messages)
    
    field=substep.get("field", "")
    value = substep.get("value", "")
    print("calling API 1")

    prompt = FILL_PROMPT.format(field=field, value=value)

    example = substep.get("example")
    if example:
        prompt += "\n\nExample qualified answers are: " + example

    filled_value = api_call(prompt, recent_messages)
    print("Filled value:", filled_value)

    # extract just the number
    if "numbers" in value.lower():
        filled_value = extract_number(filled_value)
    if not filled_value or "clarification_required" in filled_value:
        # ask for clarification if no number found
        # print("calling API 2")
        # print("field:", field)
        # print("value:", value)
        # clarification_question = api_call(
        #     FILL_CLARIFICATION_PROMPT.format(field=field, value=value),
        #     messages
        # )
        return {
            "intent": intent,
            "action": "",
            "botMessage": "Sorry, I couldn’t understand very well. Could you please clarify? Feel free to type to me by the keyboard",
        }

    if "name" in value:
        print("calling API 3, PAYEE_NAME_CLEAN_PROMPT")
        # validate the payee name
        filled_value = api_call(PAYEE_NAME_CLEAN_PROMPT, messages)
    
    action = substep.get("action", [])
    # add the filled value to the action if it requires a value
    if action :
        action[0]["value"] = filled_value

    return {
        "intent": intent,
        "action": action,
        "substep_flags": {substep.get("completion_condition"): True},
    }

def checkbox_handler(substep, messages, intent, new_page_loaded):
    # 1. Ask the yes/no question if newPageLoaded
    print("====Yes/No classification handler called")
    # print("substep", substep)
    if new_page_loaded:
        print("New page loaded, asking immediate reply...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
        }

    # 2. Classify user response
    classification = api_call(YESNO_CLASSIFIER_PROMPT, messages)
    print("Classification result:", classification)
    # print("Substep:", substep)

    if classification.lower() == "yes":
        return {
            "intent": intent,
            "action": substep["options"]["yes"]["action"],
            "substep_flags": {substep.get("completion_condition"): True}
        }
    elif classification.lower() == "no":
        return {
            "intent": intent,
            "action": substep["options"]["no"]["action"],
            "substep_flags": {substep.get("completion_condition"): True}
        }
    else:
        return {
            "intent": intent,
            "botMessage": "Sorry, could you please clarify if you need to set up auto pay?"
        }

def confirmation_handler(substep, messages, intent, new_page_loaded) -> str:
    """
    Uses GPT to classify a user response as 'yes', 'no', or 'unclear'.

    Parameters:
        user_message (str): The user's message to interpret.
        action_description (str): A simple description of what is being confirmed. E.g., "send money to Bob Chen"

    Returns:
        One of: "yes", "no", "unclear"
    """
    print("===Confirmation handler called")
    if new_page_loaded:
        print("New page loaded, asking yes/no question...")
        return {
            "intent": intent,
            "botMessage": substep["immediate_reply"]
            # "botMessage": "test"
        }

    action_description = substep.get("action_description", "proceed with this action")
    # messages is a list of dicts like {"role": "user", "content": "..."}
    user_message = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        ""
    )
    prompt = f"""
You are a confirmation assistant helping to interpret user's response as 'yes', 'no', or 'unclear'.

The user is asked to confirm an action: {action_description}

User message is: "{user_message}"

Does the user confirm the action? If the user's response is clear and affirmative (such as 'confirm'), respond with "yes". If the user's response is clear and negative (such as 'cancel'), respond with "no".

Respond with exactly one word:
- yes
- no
- unclear

Do NOT explain or include any other text.
""".strip()

    result = api_call(prompt, [])
    if result.lower() == "yes":
        return {
            "intent": intent,
            "action": substep.get("action", []),
            "substep_flags": {substep.get("completion_condition"): True}
        }
    elif result.lower() == "no":
        return {
            "intent": intent,
            "botMessage": "Okay, let me know what you'd like to do instead."
        }
    else:
        return {
            "intent": intent,
            "botMessage": "Sorry, I couldn’t understand your response. Could you please confirm?"
        }


# Grace - Alex
def handle_first_incomplete_substep(substeps, substep_flags, messages, intent, new_page_loaded, state={}):
    # Check the completion conditions of each substep
    # and handle the first uncompleted substep
    print("===Handling first incomplete substep...")
    print("===substep_flags:", substep_flags)
    for name, substep in substeps.items():
        condition = substep.get("completion_condition", "") # e.g., "account_chosen"
        print("Checking condition:", condition)
        if not substep_flags.get(condition, ""): # substep_flags looks like this: {"account_chosen": True} => {"account_chosen": True, "amount_entered": False}
            print("Found first incomplete substep:", name)
            handler_type = substep.get("dynamic_handler", "")
            print("Dynamic handler type:", handler_type)
            # Need to return intent for every handler condition
            if not handler_type:
                print("No dynamic handler for current step, sending instruction directly...")
                return {
                    "intent": intent,
                    "botMessage": substep.get("immediate_reply", ""),
                    "substep_flags": {substep.get("completion_condition"): True},
                    "action": substep.get("action", ""),
                }
            elif handler_type == "yesno_handler":
                return yesno_handler(new_page_loaded, messages, substep, intent)
            elif handler_type == "classification_handler":
                return classification_handler(substep, messages, intent, new_page_loaded)
            elif handler_type == "confirmation_handler":
                return confirmation_handler(substep, messages, intent, new_page_loaded)
            elif handler_type == "selection_handler":
                return selection_handler(substep, messages, intent, new_page_loaded)
            elif handler_type == "fill_handler":
                return fill_handler(substep, messages, intent, new_page_loaded)
            elif handler_type == "checkbox_handler":
                return checkbox_handler(substep, messages, intent, new_page_loaded)


# Grace - Alex and Frank - Sam    
def handle_known_intent(intent, current_page, substep_flags, messages, new_page_loaded, state={}, assistant="grace"):
    print("==Handling known intent:", intent)
    if intent in flows and current_page in flows[intent][assistant]:
        current_step = flows[intent][assistant][current_page]
        substeps = current_step.get("substeps", {})
        print("====substeps:", substeps.keys())
        return handle_first_incomplete_substep(
            substeps, substep_flags, messages, intent, new_page_loaded, state
        )
    else:
        print("====WIP: Intent or current page not found in flows for intent:", intent, current_page)
        return None



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

# Grace - Alex
@app.post("/tutorbot")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    new_page_loaded = body.get("newPageLoaded", False)
    intent = body.get("intent") or None
    substep_flags = body.get("substep_flags", {})   # example: {"account_chosen": True}
    current_page = body.get("currentPage")    # e.g., check_transferee
    assistant = body.get("assistant", "grace")  # e.g., "grace" or "frank"

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
            follow_up = api_call(INTENT_CLARIFICATION_PROMPT, messages)
            return {
                "intent": "unknown",
                "selector": "",
                "botMessage": follow_up,
            }
    # When an intent is identified (either just identified from above block, or passed from the frontend), we need to go to the next step and send the next instruction
    return handle_known_intent(intent, current_page, substep_flags, messages, new_page_loaded, assistant=assistant)


# Frank - Sam
@app.post("/tellerbot")
async def chat(request: Request):
    print("🔔 /tellerbot hit")
    body = await request.json()
    messages = body.get("messages", [])
    intent = body.get("intent") or None
    substep_flags = body.get("substep_flags", {})   # example: {"account_chosen": True}
    current_page = body.get("currentPage")    # e.g., check_transferee
    new_page_loaded = body.get("newPageLoaded", False)
    state = body.get("state", {})  # e.g., {"account": "chequing", "amount": 100, "confirmed": None}
    assistant = body.get("assistant", "grace")  # e.g., "grace" or "frank"

    if intent in ["unknown", "null", "", "undefined", None]:
        intent = None

    print("====Intent and current_page from frontend:", intent, current_page)

    # # 🔙 GPT-powered back-intent check (runs before intent ID)
    # if wants_navigation_back(messages):
    #     return {
    #         "botMessage": "Okay — going back to the previous page.",
    #         "action": [{"action": "navigate", "value": "back"}],
    #     }

    # 1. Intent Identification
    if not intent:
        # Ask questions until intent is identified
        print("==Identifying intent...")
        intent = api_call(INTENT_PROMPT, messages)
        if intent == "clarification_required":
            print("==Intent unclear, asking for clarification...")
            follow_up = api_call(INTENT_CLARIFICATION_PROMPT, messages)
            return {
                "intent": "unknown",
                "action": "",
                "botMessage": follow_up,
            }
    
    # When an intent is identified (either just identified from above block, or passed from the frontend), we need to go to the next step and send the next instruction
    # messages is a list of dicts like {"role": "user", "content": "..."}
    user_message = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        ""
    )
    print("==User_message:", user_message)
    res = handle_known_intent(intent, current_page, substep_flags, messages, new_page_loaded, state=state, assistant=assistant)
    print("===Response from handle_known_intent:", res)
    return res

### Another endpoint to add payees

# Global list (prototype)
payees = [
    {"name": "BC Hydro", "account": "73738374622"},
    {"name": "Telus Mobile", "account": "36379939374"},
]

class Payee(BaseModel):
    name: str
    account: str

@app.post("/api/add_payee")
async def add_payee(payee: Payee):
    payees.append({"name": payee.name, "account": payee.account})
    return {"status": "success", "payees": payees}

@app.get("/api/payees")
async def list_payees():
    return {"payees": payees}


# Global list (prototype)
autopayments = []

class AutoPayment(BaseModel):
    name: str
    account: str
    enabled: bool
    amount: float
    fromAccount: str
    frequency: str
    paymentDate: str
    notify_sms: bool = False
    notify_email: bool = False

@app.post("/api/autopayments")
async def save_autopayment(ap: AutoPayment):
    autopayments.append(ap.dict())
    return {"status": "success", "autopayments": autopayments}

@app.get("/api/autopayments")
async def list_autopayments():
    return {"autopayments": autopayments}


# --- New Alerts Example ---
alerts = []  # simple in-memory list for demo

class Alert(BaseModel):
    card_type: str        # e.g. "Credit Card"
    last_digits: str      # e.g. "4126"
    threshold: float      # e.g. 100.0
    sms: bool = False
    email: bool = False
    enabled: bool = False

@app.post("/api/save_alert")
async def save_alert(alert: Alert):
    # replace existing alert for this card
    for i, a in enumerate(alerts):
        if a["card_type"] == alert.card_type and a["last_digits"] == alert.last_digits:
            alerts[i] = alert.dict()
            return {"status": "updated", "alerts": alerts}
    alerts.append(alert.dict())
    return {"status": "created", "alerts": alerts}


@app.get("/api/alerts")
async def get_alerts():
    return {"alerts": alerts}

@app.get("/api/get_alert")
async def get_alert(card_type: str, last_digits: str):
    for alert in alerts:
        if alert["card_type"] == card_type and alert["last_digits"] == last_digits:
            return {"alert": alert}
    return {"alert": None}