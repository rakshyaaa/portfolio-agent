# Portfolio AI Agent

This project implements an LLM-driven portfolio agent with two entrypoints:
1) A Streamlit UI for interactive use.
2) A FastAPI service for programmatic access.

The agent uses a local JSON data file to answer recruiter questions and
summarize portfolio details without hallucinating.

## Repository layout

- `enhanced_agent/fastapi_app.py`: FastAPI service exposing a `/portfolio/ask` endpoint.
- `enhanced_agent/streamlit_app.py`: Streamlit UI for interactive exploration and debugging.
- `enhanced_agent/portfolio_agent.py`: Core agent loop that drives tool calls and LLM reasoning.
- `enhanced_agent/agent_tools.py`: Read-only tools that expose portfolio data from JSON.
- `enhanced_agent/portfolio_data.json`: Portfolio content (single source of truth).
- `enhanced_agent/config_provider.py`: Environment-based configuration provider for API keys.
- `enhanced_agent/requirements.txt`: App-specific dependencies for the agent UI/LLM clients.
- `requirements.txt`: Top-level dependencies used elsewhere in the repo.

## How the project works

1) User asks a question (Streamlit UI or FastAPI endpoint).
2) `LLMPortfolioAgent` builds a system prompt and decides which tools to call.
3) `PortfolioTools` loads data from `portfolio_data.json`.
4) Tool results are passed back to the LLM.
5) The LLM produces a final answer strictly based on those tool results.

### Core agent flow

- `LLMPortfolioAgent.ask(...)`:
  - Maintains conversation history.
  - Sends messages to the LLM (Ollama or OpenAI).
  - Executes tool calls when requested.
  - Returns the final answer plus optional reasoning and tool call logs.

- `PortfolioTools`:
  - Exposes profile, skills, experience, and projects.
  - Supports keyword search over projects.

## Environment configuration

Configuration is loaded from environment variables (via `python-dotenv`).
`enhanced_agent/config_provider.py` is the single source of truth for required env vars.

### LLM configuration

- `OPENAI_API_KEY`: Required if using the OpenAI provider.

### FastAPI internal auth

- `INTERNAL_AUTH_TOKEN`: Required for internal auth check in FastAPI (if enabled).

## Portfolio data

Update your portfolio content in:
- `enhanced_agent/portfolio_data.json`

## Python environment setup

This repo includes local virtual environments (for example, `.venv312` and
`enhanced_agent/.venv`). Use one consistent environment for development.

Example setup on Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r enhanced_agent\requirements.txt
```

If you also need top-level dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Streamlit UI

```powershell
cd enhanced_agent
streamlit run streamlit_app.py
```

The app will prompt for a query and show tool calls + reasoning.

## Running the FastAPI service

```powershell
cd enhanced_agent
uvicorn fastapi_app:fastapi_app --reload --port 8000
```

Then visit:
- Swagger UI: `http://localhost:8000/docs`
- Endpoint: `POST http://localhost:8000/portfolio/ask`

Example request body:

```json
{
  "query": "List Rakshya's recent projects",
  "max_iterations": 5,
  "verbose": false
}
```

## Notes and assumptions

- The LLM must not fabricate data; it only uses tool results.
- Portfolio data is read-only from the agent's perspective.
