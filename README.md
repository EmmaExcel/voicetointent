# Voice to Intent

Converts spoken financial commands into structured JSON intents. It uses Whisper for transcription and an LLM for parsing. Instead of navigating forms, users can dictate a command like "Send 5k to mum" and the system extracts the necessary details.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

macOS users require portaudio for live microphone recording:

```bash
brew install portaudio
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Set LLM_PROVIDER and the corresponding API key or model name in the .env file.

## Usage

Start the server:

```bash
uvicorn main:app --reload
```

API documentation is available at http://localhost:8000/docs

CLI commands:

```bash
python cli.py extract "Send 5k to mum" --schema financial
python cli.py process recording.wav --schema financial
python cli.py listen --schema financial
```

See DEMO.md for input examples and expected outputs.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Health check |
| GET | /schemas | List registered schemas |
| POST | /transcribe | Transcribe audio file to text |
| POST | /extract | Extract JSON intent from text and schema |
| POST | /process | Full pipeline from audio file to JSON intent |
| WS | /ws/process | WebSocket for streaming audio and receiving intent |

Example request:

```bash
curl -X POST http://localhost:8000/process \
  -F "audio=@recording.wav" \
  -F "schema_name=financial"
```

Example response:

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

The confidence field contains the Whisper transcription score.

## LLM Providers

Modify LLM_PROVIDER in .env to switch between providers. No code modifications are necessary.

| Provider | .env value | Required variables |
|----------|------------|--------------------|
| Ollama (local) | ollama | OLLAMA_MODEL, OLLAMA_BASE_URL |
| LM Studio (local) | lmstudio | LMSTUDIO_MODEL, LMSTUDIO_BASE_URL |
| OpenAI | openai | OPENAI_API_KEY, OPENAI_MODEL |
| Gemini | gemini | GEMINI_API_KEY, GEMINI_MODEL |
| Anthropic | anthropic | ANTHROPIC_API_KEY, ANTHROPIC_MODEL |

## Beneficiary Data

Beneficiary records are stored in data/beneficiaries.json. In production, this data would be sourced from a user contact list or banking API.

To use a different file, run:

```bash
BENEFICIARIES_FILE=/path/to/contacts.json uvicorn main:app
```

## Docker

Build and run with Docker:

```bash
docker build -t voicetointent .
docker compose up
```

The docker-compose.yml file starts the API server and a local Ollama instance.

## Project Structure

```
voicetointent/
├── main.py
├── cli.py
├── config.py
├── core/
│   ├── transcriber.py
│   ├── mic_recorder.py
│   └── llm/
│       ├── base.py
│       ├── ollama_provider.py
│       ├── openai_provider.py
│       ├── gemini_provider.py
│       └── anthropic_provider.py
├── schemas/
│   ├── __init__.py
│   └── financial.py
├── data/
│   └── beneficiaries.json
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Tests

Run tests with pytest:

```bash
pytest tests/ -v
```
