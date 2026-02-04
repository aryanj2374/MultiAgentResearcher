**Overview**
This app takes a natural-language research question and runs a multi-agent pipeline to retrieve papers from Semantic Scholar, extract structured evidence, critique study quality, synthesize a citation-grounded answer with confidence, and verify citations before returning results in a single response.

**Architecture**
```
[Client]
   | POST /api/ask
   v
[FastAPI]
   |
   v
[Orchestrator]
   |-> RetrieverAgent -> Semantic Scholar API
   |-> ExtractorAgent -> Hugging Face LLM
   |-> QualityCriticAgent -> Hugging Face LLM
   |-> SynthesizerAgent -> Hugging Face LLM
   |-> RefereeAgent -> verification checks
   v
[Run Storage] -> backend/data/runs/<run_id>.json
```

**Setup**
1. Backend dependencies:
```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. Optional environment variables (in `backend/.env` or your shell):
```
HF_TOKEN=...
HF_MODEL=...
SEMANTIC_SCHOLAR_API_KEY=...
```
If no LLM key is set, the backend falls back to heuristic summaries and critiques.
3. Run backend:
```
cd backend
uvicorn main:app --reload
```
4. Frontend dependencies:
```
cd frontend
npm install
```
5. Run frontend:
```
cd frontend
npm run dev
```

**Example Questions**
- Does mindfulness improve sleep quality in adults?
- Do GLP-1 agonists reduce cardiovascular risk in patients with type 2 diabetes?
- What is the evidence that urban green space reduces anxiety?

**Limitations / Next Steps**
- Abstract-only extraction can miss methods, results, and bias details.
- The referee pass is deterministic; future work could include a second LLM judge.
- Add full-text retrieval, PDF parsing, and stronger citation disambiguation.
- Add async streaming updates for each agent stage in the UI.
