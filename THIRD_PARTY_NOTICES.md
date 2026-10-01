# Third-party notices and provenance

## OpenJEV

Prompt layout and readout formulas in `src/mlx_decisions/protocol.py` are adapted
from `openjev/openjev`, `helper/shim.py` (Apache License 2.0).
The complete unchanged helper is retained as `tests/fixtures/openjev_shim.py` for
AST-isolated parity tests; it is never imported as a server or executed as a CLI.

- Source: https://huggingface.co/openjev/openjev/blob/ac97900fd034fdd7e7e536f3d4c21b836cae0750/helper/shim.py
- SHA-256: `81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a`
- Copyright: OpenJEV contributors / LoopAI, as identified by the upstream model card.
- License text: `LICENSE` (Apache-2.0).
- Changes: typed validation, text-only <=52-option contract, local raw-logit readout,
  batched scheduling. Calibration and trained text prompt preserved.

**Model weights are not included.** `openjev/openjev-MLX` weights are separately
licensed CC BY-NC 4.0, not covered by this code's Apache-2.0 license. Research and
non-commercial use only without the model owner's commercial license. Attribution:
https://huggingface.co/openjev/openjev-MLX and its `NOTICE` / license files.

## mlxfast challenge

Algorithmic inspiration, independently implemented in Python/MLX:

- Layr-Labs/mlxfast-bonsai2-27b-engine, revision
  `d038e704422d8fabf4fe0c9b05a366e362cb7d7e` (MIT, Copyright 2026 Layr Labs, Inc.).
- `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift`:
  evaluation-only/last-position prefill output narrowing.
- `Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen35FusedGateUpTests.swift`:
  gate/up quantized row-concatenation parity technique, adapted here to dense MLPs.
- `docs/bonsai2-27b-port-notes.md`: recurrent/conv-state ownership and compact carries.

No challenge source or proprietary model weights are vendored. Our selected-letter
projection is a decision-specific extension of narrowing, not a claim that the
challenge implemented this exact classifier.

## Other research and dependencies

- bnsd55/jevmlx (MIT), revision `7e0d746081b88412ccd7d84a5ffdcf9d61b36904`:
  design reference for prefix branching, selected heads, parity and bounded serving.
  No source copied.
- Bespoke Labs Nimble: alternative decision-model contract and parallel scoring
  research; no code or weights copied.
- MLX / MLX-LM (Apple Inc., MIT), Transformers, Hugging Face Hub, FastAPI, Uvicorn,
  and Pydantic remain independently licensed dependencies; see their distributions.
