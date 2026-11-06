import os
import json
import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Environment Variables (Task 3 Requirement)
MODEL_URI = os.environ.get("MODEL_URI", "models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 64))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0")
POD_NAME = os.environ.get("HOSTNAME", socket.gethostname())

ALLOWED_KEYS = {"intent", "date", "time", "place"}
ALLOWED_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query", "recommendation_events"
}

tokenizer = None
model = None
is_model_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, is_model_ready
    try:
        print(f"Loading model from {MODEL_URI}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_URI)
        model = AutoModelForCausalLM.from_pretrained(MODEL_URI)
        model.eval()
        is_model_ready = True
        print(f"Model {MODEL_VERSION} loaded successfully on {POD_NAME}!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        is_model_ready = False
    yield

app = FastAPI(title="Command Parser Service", lifespan=lifespan)

class ParseRequest(BaseModel):
    text: str

@app.get("/healthz")
def healthz():
    return Response(content="OK", status_code=status.HTTP_200_OK)

@app.get("/readyz")
def readyz():
    if is_model_ready:
        return Response(content="READY", status_code=status.HTTP_200_OK)
    return Response(content="MODEL_NOT_READY", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.post("/parse")
def parse_command(req: ParseRequest, response: Response):
    response.headers["X-Model-Version"] = MODEL_VERSION
    response.headers["X-Served-By"] = POD_NAME

    if not is_model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"error": "service_unavailable"}

    prompt = f"Command: {req.text}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_json_part = full_output.split("JSON:")[-1].strip()

    # Fault Injection check (Task 3 requirement)
    if FAULT_INJECT == "1":
        generated_json_part = "not json"

    # Schema Validation
    try:
        parsed = json.loads(generated_json_part)
        if not isinstance(parsed, dict) or set(parsed.keys()) != ALLOWED_KEYS:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": "invalid_model_output"}
        if parsed["intent"] not in ALLOWED_INTENTS:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error": "invalid_model_output"}
    except Exception:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "invalid_model_output"}

    response.status_code = status.HTTP_200_OK
    return parsed