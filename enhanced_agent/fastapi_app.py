from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config_provider import EnvConfigProvider
from llm_fundraising_agent import LLMPortfolioAgent
from ollama import Client as OllamaClient
from openai import OpenAI

fastapi_app = FastAPI()


# CORS configuration
origins = [
    "http://localhost:4200",  # Angular dev server
    "http://127.0.0.1:4200",
]

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


openai_api_key = EnvConfigProvider().get_openai_api_key()


def get_llm_client(llm_type: str, api_key: str = None):
    """Initialize LLM client"""
    if llm_type == "Ollama":
        try:
            client = OllamaClient(host="http://localhost:11434")
            client.list()  # Test connection
            return client, None
        except ImportError:
            return None, "Ollama package not installed. Run: pip install ollama"
        except Exception as e:
            return None, f"Ollama connection failed: {str(e)}"
    elif llm_type == "OpenAI":
        try:
            if not api_key:
                return None, "OpenAI API key required"
            client = OpenAI(api_key=api_key)
            return client, None
        except ImportError:
            return None, "OpenAI package not installed. Run: pip install openai"
        except Exception as e:
            return None, f"OpenAI initialization failed: {str(e)}"

    else:
        raise ValueError("Unsupported LLM type")


llm_client, llm_error = get_llm_client(
    llm_type="OpenAI", api_key=openai_api_key)
if llm_error or llm_client is None:
    raise RuntimeError(f"Failed to initialize LLM client: {llm_error}")

agent = LLMPortfolioAgent(
    llm_client=llm_client,
    model="gpt-3.5-turbo",
    provider="openai",
)


# Pydantic request model
class AskRequest(BaseModel):
    query: str
    max_iterations: int = 5
    verbose: bool = False


# Pydantic response model
class AskResponse(BaseModel):
    answer: str


INTERNAL_TOKEN = EnvConfigProvider().get_internal_auth_token()


# Secure FastAPI with an internal header -- so that only spring boot can call it
async def verify_internal_auth(x_internal_auth: str = Header(None)):
    if x_internal_auth != INTERNAL_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized"
        )


# @fastapi_app.post("/portfolio/ask", response_model=AskResponse, dependencies=[Depends(verify_internal_auth)])
@fastapi_app.post("/portfolio/ask", response_model=AskResponse)
async def ask_agent(req: AskRequest):
    try:
        resp = agent.ask(
            req.query,
            max_iterations=req.max_iterations,
            verbose=req.verbose,
        )

        return AskResponse(answer=resp["answer"])
    except Exception as e:
        # Prevent leaking full tracebacks in the response
        raise HTTPException(status_code=500, detail=str(e))
