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

# New OpenAI client for Azure
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE")
)

flows = {
    "e_transfer": [
        {"desc": "Click the 'Pay & Transfer' tab", "selector": "#nav-transfer"},
        {"desc": "Enter the payee name", "selector": "#payee-name"},
        {"desc": "Enter the amount", "selector": "#amount"},
        {"desc": "Click the 'Send' button", "selector": "#send-button"}
    ]
}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])  # full chat history
    step_index = body.get("stepIndex", 0)

    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful banking assistant. "
            "Your job is to help the user identify their goal (one of: e_transfer, pay_bill, check_balance). "
            "Ask clarifying questions if needed. Once you're sure, respond only with the intent name like 'e_transfer'."
        )
    }

    # Azure deployment name (e.g., "gpt-4" or your specific deployment name)
    deployment_name = "gpt-4"

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[system_msg] + messages
    )

    reply = response.choices[0].message.content.strip().lower()

    if reply in flows:
        step = flows[reply][step_index] if step_index < len(flows[reply]) else {"desc": "Task complete", "selector": ""}
        return {
            "intent": reply,
            "step": step,
            "stepIndex": step_index,
            "botMessage": step["desc"]
        }
    else:
        return {
            "intent": "unknown",
            "step": {},
            "stepIndex": 0,
            "botMessage": reply
        }
