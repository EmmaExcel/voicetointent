# Voice-to-Intent

Takes a spoken financial command — either typed text, an audio file, or live mic input — and returns a structured JSON intent. Runs through Whisper for transcription, then an LLM to parse the intent.

The idea is simple: instead of making users navigate a transfer form, they just say *"Send 5k to mum"* and the system figures out the rest.

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

macOS users also need portaudio for live mic recording:

```bash
brew install portaudio
```

### Configure

```bash
cp .env.example .env
```

Open `.env` and set `LLM_PROVIDER` along with the API key or model name for whichever provider you're using.

### Start the server

```bash
uvicorn main:app --reload
```

Swagger docs at http://localhost:8000/docs

### Or use the CLI

```bash
python cli.py extract "Send 5k to mum" --schema financial

python cli.py process recording.wav --schema financial

python cli.py listen --schema financial
```

Check [DEMO.md](DEMO.md) for example inputs and what the output looks like.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/schemas` | List registered schemas |
| `POST` | `/transcribe` | Audio file → transcript + confidence |
| `POST` | `/extract` | Text + schema → JSON intent |
| `POST` | `/process` | Audio file + schema → full pipeline |
| `WS` | `/ws/process` | WebSocket: stream audio, get intent back |

### Example

```bash
curl -X POST http://localhost:8000/process \
  -F "audio=@recording.wav" \
  -F "schema_name=financial"
```

```json
{
  "transcript": "Send 5k to mum",
  "confidence": 0.91,
  "schema_name": "financial",
  "intent": {
    "intent": "send_money",
    "amount": 5000,
    "currency": "NGN",
    "recipient": {
      "id": "ben_001",
      "full_name": "Jane Doe",
      "account_number": "0123456789",
      "bank_name": "Guaranty Trust Bank (GTB)"
    }
  }
}
```

`confidence` is Whisper's transcription score. Worth gating on this before actually executing a transaction.

---

## LLM Providers

Switch providers by changing `LLM_PROVIDER` in `.env` — no code changes needed.

| Provider | `.env` value | Required variables |
|----------|--------------|-------------------|
| Ollama (local) | `ollama` | `OLLAMA_MODEL`, `OLLAMA_BASE_URL` |
| LM Studio (local) | `lmstudio` | `LMSTUDIO_MODEL`, `LMSTUDIO_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Gemini | `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |

---

## Beneficiary Data

Contacts live in `data/beneficiaries.json`. It's kept separate from the source code on purpose — in a real app you'd generate this from the user's contact list or pull it from a banking API.

To point it at a different file:

```bash
BENEFICIARIES_FILE=/path/to/contacts.json uvicorn main:app
```

---

## Docker

```bash
docker build -t voicetointent .

docker compose up
```

`docker-compose.yml` spins up the API server and an Ollama instance together, so everything runs locally.

---

## Project Structure

```
voicetointent/
├── main.py                  # FastAPI app
├── cli.py                   # Typer CLI
├── config.py                # Env config and provider factory
├── core/
│   ├── transcriber.py       # faster-whisper wrapper
│   ├── mic_recorder.py      # live mic recording with VAD
│   └── llm/
│       ├── base.py          # abstract provider + system prompt routing
│       ├── ollama_provider.py
│       ├── openai_provider.py
│       ├── gemini_provider.py
│       └── anthropic_provider.py
├── schemas/
│   ├── __init__.py          # schema registry
│   └── financial.py         # TransactionIntent schema
├── data/
│   └── beneficiaries.json
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_transcriber.py
│   ├── test_llm_base.py
│   ├── test_providers.py
│   ├── test_api.py
│   └── test_config.py
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Tests

```bash
pytest tests/ -v
```
