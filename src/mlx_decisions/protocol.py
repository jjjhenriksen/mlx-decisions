"""OpenJEV text-lane prompt and calibrated readout contract.

Prompt layout and formulas adapted from openjev/openjev helper/shim.py
(Apache-2.0). See THIRD_PARTY_NOTICES.md. Never replace with a generic
classification prompt: this model was tuned for the default state-first layout.
Rubric-first is an opt-in experimental reordering, not a calibrated equivalent.
"""

import json
import math
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LETTERS = string.ascii_uppercase + string.ascii_lowercase
MODEL_ID = "openjev/openjev-MLX"
MODEL_REVISION = "a9dcc20aa827a6c7eae478f6ebb3b255bb135451"
TEMPERATURE = 0.85
NOUL_TEMPERATURE = 1.829074


def description(value: Any) -> str:
    return (
        ""
        if value is None
        else value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "score", "noul"]
    instructions: str | dict[str, Any]
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def check(self):
        c = self.criteria
        if self.type == "choice" and (not isinstance(c, dict) or not 1 <= len(c) <= 52):
            raise ValueError("choice.criteria must contain 1..52 options")
        if self.type == "score" and (not isinstance(c, list) or not 2 <= len(c) <= 52):
            raise ValueError("score.criteria must contain 2..52 ordered levels")
        if (
            self.type == "noul"
            and c is not None
            and (not isinstance(c, dict) or set(c) - {"true", "false"})
        ):
            raise ValueError("noul.criteria accepts only true/false descriptions")
        # Validate nested values now, not in the GPU worker.
        json.dumps(self.model_dump(), allow_nan=False)
        return self

    def options(self) -> list[tuple[str, str]]:
        if self.type == "choice":
            return [(k, description(v)) for k, v in self.criteria.items()]
        if self.type == "score":
            return [(str(i), description(v)) for i, v in enumerate(self.criteria)]
        c = self.criteria or {}
        return [
            ("yes", description(c.get("true")) or "The statement is true."),
            ("no", description(c.get("false")) or "The statement is false."),
        ]

    def rubric(self) -> str:
        instructions = str(self.instructions)  # trained pyrepr mode for dict instructions
        if self.type == "score":
            instructions += " Rate along the ordered levels below (lowest first)."
        lines = "\n".join(f"[{LETTERS[i]}] {k}: {d}" for i, (k, d) in enumerate(self.options()))
        return f"Question: {instructions}\nOptions:\n{lines}"

    def prefix(self, state: str, *, rubric_first: bool = False) -> str:
        return self.rubric() if rubric_first else f"State:\n{state}"

    def prompt(self, state: str, *, rubric_first: bool = False) -> str:
        state_block, rubric = f"State:\n{state}", self.rubric()
        blocks = (rubric, state_block) if rubric_first else (state_block, rubric)
        return "\n\n".join((*blocks, "Answer with the letter of the best option only."))

    def answer(self, logits: list[float]) -> dict:
        opts = self.options()
        if len(logits) != len(opts) or not all(math.isfinite(x) for x in logits):
            raise ValueError("missing or nonfinite candidate logits")
        z = [v / TEMPERATURE for v in logits]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        p = [v / sum(e) for v in e]
        winner = max(range(len(p)), key=p.__getitem__)
        if self.type == "noul":
            py = min(max(p[0], 1e-4), 1 - 1e-4)
            z = math.log(py / (1 - py)) / NOUL_TEMPERATURE
            return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}
        probabilities = {k: round(v, 4) for (k, _), v in zip(opts, p)}
        if self.type == "choice":
            confidence = 1.0 if len(p) == 1 else max(0.0, (max(p) - 1 / len(p)) / (1 - 1 / len(p)))
            return {
                "type": "choice",
                "choice": opts[winner][0],
                "probabilities": probabilities,
                "confidence": round(confidence, 4),
            }
        center = (len(p) - 1) / 2
        umad = sum(abs(i - center) for i in range(len(p))) / len(p)
        confidence = max(0.0, 1 - sum(v * abs(i - winner) for i, v in enumerate(p)) / umad)
        return {
            "type": "score",
            "score": round(sum(i * v for i, v in enumerate(p)), 4),
            "legend": {str(i): v for i, v in enumerate(self.criteria)},
            "probabilities": probabilities,
            "confidence": round(confidence, 4),
        }


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    state: str | dict[str, Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def text_only(self):
        if isinstance(self.state, dict) and any(k in self.state for k in ("image", "screenshot")):
            raise ValueError("OpenJEV-MLX is text-only; image/screenshot input is unsupported")
        json.dumps(self.state, allow_nan=False)
        return self

    def state_text(self) -> str:
        return (
            self.state
            if isinstance(self.state, str)
            else json.dumps(self.state, ensure_ascii=False)
        )
