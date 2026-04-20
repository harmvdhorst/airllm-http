# FastAPI HTTP Wrapper for airllm

This project implements a FastAPI HTTP wrapper around the airllm model, designed to keep the model loaded between requests, allowing for per-request generation options.

## Endpoints

- `GET /health`: Check if the service is running.
- `GET /ready`: Check if the service is ready to handle requests.
- `POST /v1/chat/completions`: OpenAI-compatible completion endpoint, accepting:
  - `model`: model override (optional)
  - `messages`: user-provided messages
  - `temperature`: sampling temperature
  - `top_p`: nucleus sampling
  - `max_tokens`: maximum number of tokens to generate
  - `stop`: stop sequences
  - `stream`: streaming enabled (false only)
  - Additional airllm options can be passed through.

## Model Management

Singleton ModelManager with async lock to prevent double-loading of the model. The model 'garage-bAInd/Platypus2-70B-instruct' will be loaded on startup or the first request and reused across requests. There is also an optional endpoint `POST /unload` to free up the model from memory.

## Project Structure

- **.env.example**: Environment variable example.
- **Dockerfile**: Docker configuration for deployment.
- **README.md**: Project documentation.
- **requirements.txt**: Python package dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Run the application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Example CURL commands

Check service health:
```bash
curl http://localhost:8000/health
```

Check service readiness:
```bash
curl http://localhost:8000/ready
```

Request a chat completion:
```bash
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello!"}],"temperature":0.7}'
```
