"""Local JSON question-answering backend for the MPKG SQLite graph."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from graph_rag import GraphRAG, GraphRetriever, QwenGenerator


def create_service(args):
    retriever = GraphRetriever(args.db, max_documents=args.max_documents)
    generator = QwenGenerator(model_name=args.model, offline=args.offline,
                              max_new_tokens=args.max_new_tokens)
    return GraphRAG(retriever, generator)


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok", "database": str(service.retriever.database),
                                   "model_loaded": service.generator.model is not None})
            else:
                self.respond(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/ask":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 65536:
                    raise ValueError("JSON request must be between 1 and 65536 bytes")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                question = data.get("question")
                entity = data.get("entity")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("question must be a nonempty string")
                if entity is not None and (not isinstance(entity, str) or not entity.strip()):
                    raise ValueError("entity must be a nonempty string when supplied")
                self.respond(200, service.ask(question, entity=entity))
            except (ValueError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("ask", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--db", required=True, type=Path)
        command.add_argument("--model", default="Qwen/Qwen3-1.7B")
        command.add_argument("--offline", action="store_true",
                             help="Use already cached Qwen model files")
        command.add_argument("--max-documents", type=int, default=3)
        command.add_argument("--max-new-tokens", type=int, default=320)
    ask = commands.choices["ask"]
    ask.add_argument("--question", required=True)
    ask.add_argument("--entity", help="Exact graph entity name, for disambiguation")
    serve = commands.choices["serve"]
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_documents < 1 or args.max_new_tokens < 32:
        raise ValueError("--max-documents must be positive and --max-new-tokens at least 32")
    service = create_service(args)
    if args.command == "ask":
        result = service.ask(args.question, entity=args.entity)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    server = HTTPServer((args.host, args.port), handler_for(service))
    print(f"Graph RAG backend listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
