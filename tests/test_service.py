import pytest
from fastapi.testclient import TestClient
import service.app as service_app

client = TestClient(service_app)

# Test 1: Liveness probe check
def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.text == "OK"

# Test 2: Readiness probe before model is loaded
def test_readyz_when_not_ready():
    service_app.is_model_ready = False
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.text == "MODEL_NOT_READY"

# Test 3: Fault injection returns HTTP 500 and invalid_model_output
def test_fault_injection():
    service_app.is_model_ready = True
    service_app.FAULT_INJECT = "1"
    
    # Mocking dummy tokenizer/model generation if not loaded in test env
    class DummyTokenizer:
        eos_token_id = 0
        def __call__(self, text, return_tensors):
            return {"input_ids": None}
        def decode(self, token_ids, skip_special_tokens=True):
            return "Command: dummy\nJSON: {\"intent\": \"weather_query\", \"date\": null, \"time\": null, \"place\": \"london\"}"
    
    class DummyModel:
        def generate(self, **kwargs):
            return [[0]]

    service_app.tokenizer = DummyTokenizer()
    service_app.model = DummyModel()

    response = client.post("/parse", json={"text": "what is the weather in london"})
    assert response.status_code == 500
    assert response.json() == {"error": "invalid_model_output"}