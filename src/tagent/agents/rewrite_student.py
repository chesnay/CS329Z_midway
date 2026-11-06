"""A one-shot student for Harbor.

Terminus-2 asks the model for a JSON list of terminal keystrokes. A 3B model
spends the whole trial repairing that JSON. This agent asks for the repaired
Python program, writes it to /app/solution.py, and stops. The hidden tests
stay with Harbor's verifier.
"""

import ast
import json
import re
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_PROMPT = """You are repairing your own Python 3 program. Reply with the complete program and nothing else.

The reply must be valid Python. It must read standard input and write the answer to standard output. Do not explain. Do not use Markdown.

Problem:
{instruction}

Current program:
{source}
"""

_RETRY = """That reply was not a Python program. Reply with only the complete program. Do not explain.

Current program:
{source}
"""


def extract_program(text: str) -> str | None:
    """Return a Python program from a model reply, or None.

    A fenced block wins. Otherwise the whole reply is accepted when it parses
    as a program. A reply that is only a number or a string is rejected.
    """
    fences = [block.strip("\n") for block in _FENCE.findall(text) if block.strip()]
    candidates = fences or ([text.strip()] if text.strip() else [])
    for candidate in reversed(candidates):
        if _is_python_program(candidate):
            return candidate if candidate.endswith("\n") else candidate + "\n"
    return None


def _is_python_program(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Constant):
            return True
    return False


class RewriteStudent(BaseAgent):
    """Ask once for a whole program, write it over /app/solution.py, then stop."""

    def __init__(self, *args, temperature: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)

    @staticmethod
    @override
    def name() -> str:
        return "rewrite-student"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("rewrite-student requires --model")

        source = await self._read_starter(environment)
        llm = LiteLLM(model_name=self.model_name, temperature=self.temperature)
        prompt = _PROMPT.format(instruction=instruction.strip(), source=source)
        transcript = []
        program = None
        for attempt in (1, 2):
            response = await llm.call(prompt, logging_path=self.logs_dir / f"call_{attempt}.json")
            self._add_usage(context, response)
            program = extract_program(response.content or "")
            transcript.append(
                {
                    "attempt": attempt,
                    "temperature": self.temperature,
                    "extracted": program is not None,
                    "reply": response.content,
                }
            )
            if program is not None:
                break
            prompt = _RETRY.format(source=source)

        (self.logs_dir / "rewrite.json").write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if program is None:
            self.logger.warning("The model did not return a Python program. The starter was left in place.")
            return

        rewritten = self.logs_dir / "rewritten_solution.py"
        rewritten.write_text(program, encoding="utf-8")
        await environment.upload_file(rewritten, "/app/solution.py")

    async def _read_starter(self, environment: BaseEnvironment) -> str:
        result = await environment.exec("cat /app/solution.py", timeout_sec=30)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Could not read /app/solution.py: {detail}")
        return result.stdout

    @staticmethod
    def _add_usage(context: AgentContext, response) -> None:
        usage = response.usage
        if usage is None:
            return
        context.n_input_tokens = (context.n_input_tokens or 0) + usage.prompt_tokens
        context.n_output_tokens = (context.n_output_tokens or 0) + usage.completion_tokens
        context.n_cache_tokens = (context.n_cache_tokens or 0) + usage.cache_tokens
        context.cost_usd = (context.cost_usd or 0) + usage.cost_usd
