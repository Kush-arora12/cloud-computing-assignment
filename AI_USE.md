# AI Use Disclosure

As per assignment guidelines, AI tools were utilized with full transparency and empirical verification.

### Tools Used:
1. **Phi-3.5-mini-instruct (via Ollama)**:
   - **Role**: Teacher model for Sequence-Level Knowledge Distillation.
   - **Application**: Zero-shot command parsing into 4-key JSON (`intent`, `date`, `time`, `place`) with temperature 0 and seed 42.
   - **Verification**: Outputs filtered and strictly validated against `schema.json` via `worker/clean.py` rejecting hallucinated slots, non-allowed intents, and syntax errors.

2. **LLM Assistance (Gemini)**:
   - **Role**: Architecture drafting, boilerplate code generation, and networking calculations.
   - **Application**:
     - Designing the VLSM subnet allocation and stateless NACL rules for Task 5.
     - Structuring the FastAPI inference service, health probes, and Kubernetes manifests.
   - **Verification**:
     - All NACL rules were empirically verified using `tools/nacl_sim.py` against sample packets.
     - Subnet mathematics independently cross-verified using standard host allocation formulas ($2^{32-\text{prefix}} - 2$).
     - Shard outputs verified through deterministic SHA-256 checksums and automated schema scoring.