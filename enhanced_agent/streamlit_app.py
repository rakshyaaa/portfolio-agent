"""
Streamlit App for LLM-Based Portfolio Agent

This app demonstrates the autonomous LLM agent with:
- Tool call tracking
- Ollama or OpenAI support
"""

from portfolio_agent import LLMPortfolioAgent
import streamlit as st
import sys
import os
from pathlib import Path
from config_provider import EnvConfigProvider
from ollama import Client as OllamaClient
from openai import OpenAI

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
sys.path.insert(0, os.path.dirname(__file__))

DATA_PATH = Path(__file__).with_name("portfolio_data.json")


def app():
    """Main Streamlit app function"""

    # Page config
    st.set_page_config(
        page_title="AI Agent",
        layout="wide"
    )

    # Data check
    if not DATA_PATH.exists():
        st.error(f"Portfolio data file not found: {DATA_PATH}")
        st.stop()

    try:
        openai_api_key = EnvConfigProvider().get_openai_api_key()
    except EnvironmentError:
        openai_api_key = None

    @st.cache_resource
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

    # Main UI
    st.title("Portfolio AI Agent")
    st.markdown("""
    **Autonomous AI agent powered by LLM reasoning**

    The LLM decides:
    - What tools to use
    - What parameters to pass
    - When it has enough information
    - How to format the response
    """)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        st.markdown("---")

        # LLM Selection
        st.subheader("LLM Configuration")
        llm_type = st.radio(
            "Select LLM",
            ["Ollama (Local)", "OpenAI (Cloud)"],
            help="Ollama is free but requires local setup. OpenAI costs money but works immediately."
        )

        if llm_type == "Ollama (Local)":
            model_name = st.selectbox(
                "Model",
                ["gpt-oss:20b", "llama2", "mistral", "codellama"],
                help="Make sure model is pulled: ollama pull gpt-oss:20b"
            )
            api_key = None

            st.info("""
            **Setup Ollama:**
            ```bash
            ollama serve
            ollama pull gpt-oss:20b
            ```
            """)
        else:
            model_name = st.selectbox(
                "Model",
                ["gpt-3.5-turbo"]
            )
            api_key = openai_api_key

        # Agent settings
        st.markdown("---")
        st.subheader("Agent Settings")

        max_iterations = st.slider(
            "Max iterations",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum tool calling rounds"
        )

        show_reasoning = st.checkbox("Show reasoning steps", value=True)
        show_tool_calls = st.checkbox("Show tool calls", value=True)
        verbose_llm = st.checkbox("Verbose LLM output", value=False)

        st.markdown("---")

        # Example queries
        st.subheader("Example Queries")
        examples = [
            "Give me a short summary of Rakshya's background.",
            "What skills does Rakshya have?",
            "List her recent projects with links.",
            "What is her work experience?",
            "How can I contact Rakshya?"
        ]

        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.example_query = example

    # Initialize LLM client
    llm_client, error = get_llm_client(
        "Ollama" if "Ollama" in llm_type else "OpenAI",
        api_key
    )

    if error:
        st.error(error)
        st.stop()

    if not llm_client:
        st.warning("LLM client not initialized. Please configure in sidebar.")
        st.stop()

    provider = "ollama" if "Ollama" in llm_type else "openai"

    # Initialize agents
    if "llm_agent" not in st.session_state or st.session_state.get("llm_type") != llm_type:
        st.session_state.llm_agent = LLMPortfolioAgent(
            llm_client=llm_client,
            model=model_name,
            provider=provider,
            data_path=DATA_PATH,
        )
        st.session_state.llm_type = llm_type

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

                if show_reasoning and "reasoning_steps" in message:
                    with st.expander("LLM Reasoning Steps"):
                        for i, step in enumerate(message["reasoning_steps"], 1):
                            st.write(f"**{i}.** {step.get('action', step)}")
                            if isinstance(step, dict) and "result_preview" in step:
                                st.caption(
                                    f"Result: {step['result_preview']}...")

                if show_tool_calls and "tool_calls" in message:
                    with st.expander("Tools Called"):
                        for tool in message["tool_calls"]:
                            st.code(
                                f"{tool['tool']}({tool['args']})", language="python")

                if "iterations" in message:
                    st.caption(
                        f"Iterations: {message['iterations']} | Model: {message.get('model', 'N/A')}")

    # Handle example query from sidebar
    if "example_query" in st.session_state:
        prompt = st.session_state.example_query
        del st.session_state.example_query
    else:
        prompt = st.chat_input("Ask about Rakshya's portfolio...")

    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get LLM agent response
        with st.chat_message("assistant"):
            with st.spinner("LLM agent thinking..."):
                response = st.session_state.llm_agent.ask(
                    prompt,
                    max_iterations=max_iterations,
                    verbose=verbose_llm
                )
            st.markdown(response["answer"])
            if show_reasoning:
                with st.expander("LLM Reasoning Steps"):
                    for i, step in enumerate(response["reasoning_steps"], 1):
                        st.write(f"**{i}.** {step['action']}")
                        if "result_preview" in step:
                            st.caption(
                                f"Result: {step['result_preview'][:100]}...")
            if show_tool_calls:
                with st.expander("Tools Called"):
                    for tool in response["tool_calls_made"]:
                        st.code(f"{tool['tool']}({tool['args']})",
                                language="python")
            st.caption(
                f"Iterations: {response['iterations']} | Model: {response['model']}")
            # Store response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "reasoning_steps": response["reasoning_steps"],
                "tool_calls": response["tool_calls_made"],
                "iterations": response["iterations"],
                "model": response["model"]
            })

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.llm_agent.reset_conversation()
            st.rerun()

    with col2:
        st.write(f"Messages: {len(st.session_state.messages)}")

    with col3:
        st.write("Portfolio AI Agent v1.0")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        st.exception(e)
        raise e
