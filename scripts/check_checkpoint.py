"""Inspect the pinned checkpoint's tokenizer and local shard completeness.

Downloads metadata/tokenizer only. Does not load or silently substitute weights.
"""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

from mlx_decisions.protocol import LETTERS, MODEL_ID, MODEL_REVISION, DecisionRequest


def main():
    path = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.jinja", "LICENSE*", "NOTICE", "README.md"],
        )
    )
    config = json.loads((path / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    codes = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS}
    assert all(len(ids) == 1 for ids in codes.values())
    assert len({ids[0] for ids in codes.values()}) == 52
    request = DecisionRequest.model_validate_json(Path("benchmarks/example.json").read_text())
    prompts = {}
    for key, question in request.questions.items():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question.prompt(request.state_text())}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts[key] = {
            "tokens": len(ids),
            "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    files = HfApi().list_repo_tree(MODEL_ID, revision=MODEL_REVISION)
    shards = []
    for file in files:
        if file.path.endswith(".safetensors"):
            local = path / file.path
            shards.append(
                {
                    "file": file.path,
                    "expected_bytes": file.size,
                    "present_bytes": local.stat().st_size if local.exists() else 0,
                }
            )
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": config["model_type"],
        "quantization": config["quantization"],
        "letter_token_ids": codes,
        "example_prompts": prompts,
        "shards": shards,
        "all_weights_present": all(s["expected_bytes"] == s["present_bytes"] for s in shards),
        "verification_boundary": "Tokenizer/metadata check only; no full-model inference.",
    }
    output = Path("benchmarks/results/checkpoint-contract.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["model", "revision", "all_weights_present", "example_prompts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
