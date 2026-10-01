import argparse
import json
from pathlib import Path

from .protocol import MODEL_ID, MODEL_REVISION, DecisionRequest


def main():
    parser = argparse.ArgumentParser(
        description="Local OpenJEV decision inference; no generated JSON"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--full-head", action="store_true")
    parser.add_argument(
        "--fuse-gate-up", action="store_true", help="experimental; run parity benchmark first"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the pinned model without loading it")
    decide = sub.add_parser("decide")
    decide.add_argument("request", type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=3000)
    serve.add_argument("--queue-size", type=int, default=32)
    serve.add_argument("--batch-window-ms", type=float, default=4)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        print(snapshot_download(args.model, revision=args.revision))
        return
    from .engine import Engine

    def factory():
        return Engine(
            args.model,
            revision=args.revision,
            max_batch_size=args.batch_size,
            prefix_reuse=not args.no_prefix_cache,
            selected_head=not args.full_head,
            fused_gate_up=args.fuse_gate_up,
        )

    if args.command == "serve":
        import uvicorn

        from .server import create_app

        # Local-only by default and by CLI contract. Do not spawn multiple model workers.
        uvicorn.run(
            create_app(factory, capacity=args.queue_size, window_ms=args.batch_window_ms),
            host="127.0.0.1",
            port=args.port,
            workers=1,
        )
    else:
        request = DecisionRequest.model_validate_json(args.request.read_text())
        print(json.dumps(factory().decide(request), indent=2))


if __name__ == "__main__":
    main()
