"""
LLM-Based Portfolio Agent

This agent uses an LLM (Ollama or OpenAI) as the core reasoning engine.
The LLM autonomously decides:
- Which tools to use
- What parameters to pass
- How to combine information
- When it has enough data to answer

The LLM makes ALL decisions.
"""

from typing import Dict, Any
import json
from agent_tools import PortfolioTools, get_tool_definitions


class LLMPortfolioAgent:
    """
    Autonomous LLM-based portfolio agent.

    The LLM acts as the "brain" and decides everything:
    - Query understanding
    - Tool selection
    - Multi-step reasoning
    - Response generation
    """

    def __init__(self, llm_client, model: str = "gpt-3.5-turbo", provider: str = "openai", data_path=None):
        """
        Initialize LLM agent.

        Args:
            llm_client: Ollama or OpenAI client
            model: Model name to use
            provider: "ollama" or "openai"
            data_path: Optional path to portfolio_data.json
        """
        self.llm_client = llm_client
        self.model = model
        self.tools = PortfolioTools(data_path)
        self.tool_definitions = get_tool_definitions()
        self.conversation_history = []
        self.provider = provider
        # System prompt that guides the LLM's behavior
        self.system_prompt = """
You are the portfolio assistant for Rakshya Pandey.

You will receive:
- A natural language user question.
- One or more tool outputs containing JSON data about Rakshya's portfolio.

Your primary goal: answer recruiter or visitor questions using ONLY the tool outputs.

Hard rules:
1. You MUST NOT invent or guess any data, dates, skills, or project details.
2. You MUST ONLY mention information present in the tool outputs.
3. If a detail is missing, say "not available" or ask the user to clarify.
4. Do not reference external sources or assumptions.
5. Keep answers concise and factual.

Response guidelines:
- If asked for a summary, include name, tagline, and a short about summary.
- If asked about skills, group by categories as provided.
- If asked about experience, list roles with dates and key highlights.
- If asked about projects, list project name, summary, and link.
- If asked for contact info, provide the email from the data.
"""

    def _chat(self, messages, tools=None):
        """
        Backend independent chat call.

        Returns a dict with shape:
        {
            "message": {
                "role": str,
                "content": str,
                "tool_calls": optional list in OpenAI style but as plain dicts
            }
        }
        """
        if self.provider == "ollama":
            # Ollama already returns a dict with a "message" key
            resp = self.llm_client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            # Assume resp has a "message" key
            return resp

        elif self.provider == "openai":
            # OpenAI Python v1 style
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            # Normalize OpenAI Message to our expected shape
            norm = {
                "role": msg.role,
                "content": msg.content or "",
            }

            # Normalize tool calls, if any
            if getattr(msg, "tool_calls", None):
                tool_calls = []
                for tc in msg.tool_calls:
                    if tc.type != "function":
                        continue
                    args = tc.function.arguments
                    tool_calls.append(
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": args,
                            },
                        }
                    )
                if tool_calls:
                    norm["tool_calls"] = tool_calls

            usage = getattr(resp, "usage", None)
            norm_usage = None
            if usage is not None:
                norm_usage = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

            return {
                "message": norm,
                "usage": norm_usage,
            }

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def ask(
        self,
        query: str,
        max_iterations: int = 5,
        verbose: bool = False
    ) -> Dict[str, Any]:
        self.conversation_history.append({"role": "user", "content": query})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
        ]

        reasoning_steps = []
        tool_calls_made = []
        last_tool_result: str | None = None  # avoid unbound variable errors

        usage_log = []  # collect usage per LLM API call

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Use unified chat helper
            response = self._chat(messages, tools=self.tool_definitions)

            # Log token usage for this call (if available)
            llm_usage = response.get("usage")
            if llm_usage is not None:
                usage_log.append({
                    "iteration": iteration + 1,
                    **llm_usage,
                })
                if verbose:
                    print(
                        f"Token usage (iteration {iteration + 1}): "
                        f"input={llm_usage.get('input_tokens')} "
                        f"output={llm_usage.get('output_tokens')} "
                        f"total={llm_usage.get('total_tokens')}"
                    )

            if verbose:
                print(f"LLM Response: {response}")

            message = response.get("message", {})
            tool_calls = message.get("tool_calls")

            if tool_calls:
                # LLM decided to use tools
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_name = function.get("name")
                    tool_args = function.get("arguments")

                    if verbose:
                        print(f"Tool Call: {tool_name}")
                        print(f"Arguments: {tool_args}")

                    tool_result = self._execute_tool(tool_name, tool_args)
                    last_tool_result = tool_result

                    reasoning_steps.append({
                        "action": f"Called {tool_name}",
                        "arguments": tool_args,
                        "result_preview": tool_result,
                    })

                    tool_calls_made.append({
                        "tool": tool_name,
                        "args": tool_args,
                    })

                    # Ensure arguments for tool_call in messages are dict, not JSON string (For OLLAMA)
                    if isinstance(tool_args, str):
                        try:
                            parsed_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            parsed_args = {"raw_arguments": tool_args}
                    else:
                        parsed_args = tool_args

                    # Prepare arguments for the next LLM call
                    #    OpenAI wants JSON string, Ollama is fine with dict
                    if getattr(self, "provider", "ollama") == "openai":
                        arguments_for_message = (
                            tool_args
                            if isinstance(tool_args, str)
                            else json.dumps(parsed_args)
                        )
                    else:
                        arguments_for_message = parsed_args

                    tool_call_id = tool_call.get("id", f"call_{iteration}")
                    # Add assistant tool call to messages
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments_for_message,
                            },
                        }],
                    })

                    # Add tool result back for next LLM step
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result,
                    })

                # Continue loop. LLM will see new tool outputs in messages.
                continue

            # No tool calls. LLM gave final answer.
            final_answer = message.get("content", "")

            if verbose:
                print(f"Final Answer: {final_answer}")

            self.conversation_history.append({
                "role": "assistant",
                "content": final_answer,
            })

            return {
                "answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "tool_calls_made": tool_calls_made,
                "iterations": iteration + 1,
                "model": self.model,
                "tool_result": last_tool_result,
                "usage_log": usage_log,
            }

        # Fallback when reaching max_iterations
        final_response = self._chat(
            messages + [{
                "role": "system",
                "content": "Please provide your final answer based on the information gathered so far.",
            }],
            tools=None,
        )

        final_usage = final_response.get("usage")
        if final_usage is not None:
            usage_log.append({
                "iteration": max_iterations + 1,
                **final_usage,
            })
            if verbose:
                print(
                    f"Token usage (final): "
                    f"input={final_usage.get('input_tokens')} "
                    f"output={final_usage.get('output_tokens')} "
                    f"total={final_usage.get('total_tokens')}"
                )

        final_answer = final_response.get("message", {}).get(
            "content",
            "I need more information to provide a complete answer.",
        )

        self.conversation_history.append({
            "role": "assistant",
            "content": final_answer,
        })

        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "tool_calls_made": tool_calls_made,
            "iterations": max_iterations,
            "model": self.model,
            "note": "Reached max iterations",
            "usage_log": usage_log,
        }

    def _execute_tool(self, tool_name: str, arguments: Any) -> str:
        """Execute a tool and return results"""
        try:
            # Parse arguments if string
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            # Get the tool method
            tool_method = getattr(self.tools, tool_name, None)
            if not tool_method:
                return json.dumps({"error": f"Tool {tool_name} not found"})

            # Call the tool
            result = tool_method(**args)
            return result

        except Exception as e:
            return json.dumps({"error": f"Error executing {tool_name}: {str(e)}"})

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []


if __name__ == "__main__":
    print("LLMPortfolioAgent is ready. Configure a client to run it.")
