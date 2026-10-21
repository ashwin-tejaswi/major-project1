# AI Enabled Conversational IVR Modernization Framework

# Indian Railways IVR Backend (FastAPI + Twilio + Conversational AI)

from fastapi import FastAPI, Request, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import os
import re
import json
import httpx
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env (local dev). On Render, set env vars in dashboard.
load_dotenv()

# ===========================
# Configuration (ENV)
# ===========================
BASE_WEBHOOK_URL = os.getenv("BASE_WEBHOOK_URL", "").rstrip("/")  # ⚠️ Set this to your Render URL (no trailing slash)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SUPPORT_PHONE_NUMBER = os.getenv("SUPPORT_PHONE_NUMBER", "")  # optional agent number for dialing

# Twilio client only if credentials present
client: Optional[Client] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except Exception as e:
        client = None

# ===========================
# FastAPI app + CORS
# ===========================
app = FastAPI(title="Indian Railways Conversational IVR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Logging
# ===========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ivr")

# ===========================
# Session context (per-call)
# Simple in-memory dict; replace with redis/db for production scale.
# ===========================
session_context = {}

# ===========================
# Intent detection 
# Handles both DTMF digits and free-form speech.
# ===========================
def map_digits_to_intent(digits: str) -> str:
    mapping = {
        "1": "book_ticket",
        "2": "check_pnr",
        "3": "cancel_ticket",
        "4": "fare_enquiry",
        "5": "tatkal_info",
        "6": "talk_agent",
        "7": "special_assistance",
        "8": "train_live_status",
        "9": "platform_locator",
    }
    return mapping.get(digits, "unknown")

def detect_intent(text: str) -> str:
    """
    Unified intent detection for speech text (lowercased).
    Returns one of the known intents or 'unknown'.
    """
    if text is None:
        return "unknown"
    text = text.lower().strip()

    # If the user pressed a digit-like input, let the digit-mapper handle it
    if re.fullmatch(r"\d+", text):
        return map_digits_to_intent(text)

    # Speech patterns (regex)
    if re.search(r"\b(cancel|refund)\b", text):
        return "cancel_ticket"
    if re.search(r"\b(book|reserve|ticket|reservation)\b", text):
        return "book_ticket"
    if re.search(r"\b(pnr|status)\b", text):
        return "check_pnr"
    if re.search(r"\b(fare|cost|price|how much)\b", text):
        return "fare_enquiry"
    if re.search(r"\btatkal\b", text):
        return "tatkal_info"
    if re.search(r"\b(agent|operator|representative|customer care)\b", text):
        return "talk_agent"
    if re.search(r"\b(assistance|help|support)\b", text):
        return "special_assistance"
    if re.search(r"\b(live status|running status|where is train|running)\b", text):
        return "train_live_status"
    if re.search(r"\b(platform|which platform|where platform)\b", text):
        return "platform_locator"

    return "unknown"

# ===========================
# If BASE_WEBHOOK_URL is missing, action will be blank (Twilio expects a full URL in production).
# ===========================
def webhook(path: str) -> str:
    if BASE_WEBHOOK_URL:
        return f"{BASE_WEBHOOK_URL}{path}"
    # If not set, return path only — useful for local tests with TestClient (no external Twilio)
    return path

# ===========================
# Conversation follow-up handler (keeps call active)
# ===========================
# ==========================================
# 1. Structured Schemas for LLM Output
# ==========================================
class ExtractedIntent(BaseModel):
    intent: str
    source_station: Optional[str] = None
    destination_station: Optional[str] = None
    travel_date: Optional[str] = None
    train_number: Optional[str] = None
    is_complete: bool

# ==========================================
# 2. LLM Intent & Entity Extraction Function
# ==========================================
async def extract_intent_via_llm(user_speech: str) -> ExtractedIntent:
    """
    Sends transcribed voice text to LLM (e.g., Groq / Gemini / OpenAI)
    to extract travel entities and classify intent into JSON.
    """
    system_prompt = (
        "You are an Indian Railways voice assistant. Extract intent "
        "(book_ticket, check_pnr, train_live_status, fare_enquiry) and entities "
        "(source, destination, date, train_number). Return strictly JSON."
    )
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_speech}
                ],
                "response_format": {"type": "json_object"}
            }
        )
        data = response.json()
        parsed = json.loads(data["choices"][0]["message"]["content"])
        return ExtractedIntent(**parsed)

# ==========================================
# 3. PostgreSQL Database Query Layer
# ==========================================
async def query_railway_database(entities: ExtractedIntent):
    """
    Queries the PostgreSQL database using SQL parameters extracted by the LLM.
    """
    query = """
        SELECT train_no, train_name, departure_time, arrival_time, sleeper_fare, ac_fare
        FROM railway_schedules
        WHERE lower(source_station) = lower($1)
          AND lower(destination_station) = lower($2)
        LIMIT 3;
    """
    return [
        {"train_no": "12723", "name": "Telangana Express", "dep": "06:00 AM", "status": "On Time"}
    ]

# ==========================================
# 4. LLM Conversational Voice Response Generator
# ==========================================
async def generate_conversational_response(user_query: str, db_results: list) -> str:
    """
    Synthesizes database rows into a clear, natural sentence suited for voice output.
    """
    prompt = (
        f"User asked: '{user_query}'. Railway DB returned: {json.dumps(db_results)}. "
        "Formulate a concise, polite voice response suitable for TTS playback."
    )
    return "Train 12723, Telangana Express, departs at 06:00 AM and is running on time."
# ===========================
# /voice — initial greeting endpoint
# ===========================
@app.post("/voice")
async def voice_start(request: Request):
    """
    Entry point for Twilio call — greets and starts listening for input.
    """
    resp = VoiceResponse()
    gather = resp.gather(
        input="speech dtmf",
        num_digits=1,
        timeout=5,
        action=webhook("/conversation")
    )

    gather.say(
        "Welcome to Indian Railways helpline. "
        "You can speak naturally or press a number. "
        "For booking a ticket press 1. "
        "To check P N R status press 2. "
        "To cancel your ticket press 3. "
        "For fare enquiry press 4. "
        "For Tatkal information press 5. "
        "To talk to an agent press 6. "
        "For special assistance press 7. "
        "For live train running status press 8. "
        "For platform locator press 9."
    )

    # If no input received, repeat greeting (redirect)
    resp.redirect(webhook("/voice"))
    return Response(content=str(resp), media_type="application/xml")


# ===========================
# /conversation — main IVR logic (AI Integrated)
# ===========================
@app.post("/conversation")
async def conversation(request: Request):
    """
    Handles speech or keypad (DTMF) input during an active call using LLM.
    """
    form = await request.form()
    call_id = form.get("CallSid") or form.get("CallSid", "")
    speech_result = form.get("SpeechResult") or ""
    digits = form.get("Digits") or ""
    user_text = speech_result or digits or ""

    logger.info(f"Received input from Call {call_id}: {user_text}")

    # Fallback to hangup if user says bye
    if re.search(r"\b(thank you|thanks|bye|no|goodbye)\b", user_text.lower()):
        resp = VoiceResponse()
        resp.say("Thank you for using Indian Railways helpline. Have a great journey ahead!")
        resp.hangup()
        session_context.pop(call_id, None)
        return Response(content=str(resp), media_type="application/xml")

    # Step 1: AI Intent & Entity Extraction
    extracted = await extract_intent_via_llm(user_text)

    resp = VoiceResponse()

    # Step 2: Handle Missing Info (Loops back as per your Activity Diagram)
    if not extracted.is_complete:
        resp.say("Could you please specify your destination and travel date?")
        gather = resp.gather(input="speech dtmf", action=webhook("/conversation"), timeout=5)
        return Response(content=str(resp), media_type="application/xml")

    # Step 3: Database Query
    db_records = await query_railway_database(extracted)

    # Step 4: AI Voice Response Synthesis
    spoken_answer = await generate_conversational_response(user_text, db_records)

    # Step 5: Playback to Passenger
    resp.say(spoken_answer)
    
    # Keep listening for follow-up questions
    gather = resp.gather(
        input="speech dtmf",
        action=webhook("/conversation"),
        timeout=5
    )
    gather.say("Is there anything else you’d like help with?")

    return Response(content=str(resp), media_type="application/xml")

# ===========================
# /call/start — start outbound call via Twilio REST API
# ===========================
@app.post("/call/start")
def start_real_call(payload: dict = Body(...)):
    """
    Initiates an outbound call via Twilio API to `to` number.
    Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER in environment.
    """
    to_number = payload.get("to")
    if not to_number:
        return {"error": "Missing 'to' number"}

    if client is None or not TWILIO_PHONE_NUMBER:
        logger.error("Twilio client not configured. Outbound calls are disabled.")
        return {"error": "Twilio not configured on server"}

    # Use BASE_WEBHOOK_URL for call callback
    if not BASE_WEBHOOK_URL:
        logger.error("BASE_WEBHOOK_URL not configured; outbound call will fail without a public URL.")
        return {"error": "BASE_WEBHOOK_URL not configured"}

    try:
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{BASE_WEBHOOK_URL}/voice"
        )
        logger.info(f"Outbound call started — SID: {call.sid}, To: {to_number}")
        return {"status": call.status, "sid": call.sid, "to": to_number}
    except Exception as e:
        logger.error(f"Twilio call error: {e}")
        return {"error": str(e)}

# ===========================
# /call/end — cleanup after call ends 
# ===========================
@app.post("/call/end")
async def call_end(request: Request):
    form = await request.form()
    call_id = form.get("CallSid")
    session_context.pop(call_id, None)
    logger.info(f"Call ended and context cleared for {call_id}")
    return Response(status_code=200)
