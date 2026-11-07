"""A page for cleaning the replies the shells record before they become training data:
every drawing on its staff with its label and the model's answer, a drawing dropped or
given the label it should have had, and the replies kept written to a new file for
``musicset symbols --strokes``::

    uv run python scripts/replies_viewer.py models/battuta/replies/*.jsonl \\
        --output models/battuta/replies/clean.jsonl

The page is served on 127.0.0.1 only and opened in the browser. Several files are merged
in order, a reply that appears twice (a file shared twice) kept once; the input files are
never written. *Save* writes ``--output``, the replies kept, each line as it came but for
a new label, and ``<output>.review.json``, the decisions by reply, which the next session
over the same files starts from. Keys: arrows to move, ``x`` to drop or keep a drawing,
``Enter`` to relabel it, ``Ctrl+S`` to save.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from granita.config import BATTUTA_CLASSES, CLASS_GROUPS, KAKIGORI_CLASSES

PAGE = Path(__file__).with_name("replies_viewer.html")
CATALOG = KAKIGORI_CLASSES
"""Every class a reply may carry: musicset's whole catalog (kakigori's set)."""


@dataclass(frozen=True, slots=True)
class Reply:
    key: str
    """A hash of the line: what a decision is filed under across sessions."""
    file: str
    line: int
    raw: str
    record: dict[str, Any]


def load_replies(paths: list[Path]) -> tuple[list[Reply], list[dict[str, object]], int]:
    """The replies of ``paths`` in order, the lines that are not one, and how many lines
    repeated a reply already read."""
    replies: list[Reply] = []
    broken: list[dict[str, object]] = []
    seen: set[str] = set()
    repeats = 0
    for path in paths:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "label" not in record or "strokes" not in record:
                    raise ValueError("not a reply: no label or strokes")
            except ValueError as e:
                broken.append({"file": path.name, "line": n, "error": str(e)})
                continue
            key = hashlib.blake2b(line.encode("utf-8"), digest_size=12).hexdigest()
            if key in seen:
                repeats += 1
                continue
            seen.add(key)
            replies.append(Reply(key, path.name, n, line, record))
    return replies, broken, repeats


def flags(record: dict[str, Any]) -> list[str]:
    """What deserves a second look in a reply (the page adds "model disagrees", which
    follows the label as it is relabelled)."""
    out = []
    label = record.get("label")
    if label not in CATALOG:
        out.append("unknown label")
    elif label not in BATTUTA_CLASSES:
        out.append("not a battuta class")
    if not any(record.get("strokes") or []):
        out.append("no strokes")
    if not isinstance(record.get("staff"), dict):
        out.append("no staff")
    return out


def cleaned_lines(
    replies: list[Reply], decisions: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, int]]:
    """The lines of the replies kept, in order, a relabelled one rewritten with its new
    label; and how many were kept, dropped and relabelled."""
    lines: list[str] = []
    counts = {"kept": 0, "dropped": 0, "relabelled": 0}
    for r in replies:
        d = decisions.get(r.key, {})
        if d.get("drop"):
            counts["dropped"] += 1
            continue
        counts["kept"] += 1
        label = d.get("label")
        if label and label != r.record.get("label"):
            counts["relabelled"] += 1
            record = {**r.record, "label": label}
            lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        else:
            lines.append(r.raw)
    return lines, counts


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Review:
    """The replies under review, the decisions taken on them, and where they go."""

    def __init__(self, inputs: list[Path], output: Path) -> None:
        self.inputs = inputs
        self.output = output
        self.review = output.with_name(output.name + ".review.json")
        self.replies, self.broken, self.repeats = load_replies(inputs)
        self.decisions: dict[str, dict[str, Any]] = (
            json.loads(self.review.read_text(encoding="utf-8")).get("decisions", {})
            if self.review.is_file()
            else {}
        )

    def payload(self) -> dict[str, Any]:
        groups: dict[str, list[str]] = {}
        for name in CATALOG:
            groups.setdefault(CLASS_GROUPS.get(name, "other"), []).append(name)
        replies = []
        for i, r in enumerate(self.replies):
            rec, d = r.record, self.decisions.get(r.key, {})
            replies.append(
                {
                    "id": i,
                    "file": r.file,
                    "line": r.line,
                    "label": rec.get("label"),
                    "predicted": rec.get("predicted"),
                    "strokes": [[p[:2] for p in s] for s in rec.get("strokes") or []],
                    "staff": rec.get("staff"),
                    "staff_space": rec.get("staff_space"),
                    "notes": rec.get("notes") or [],
                    "device": rec.get("device"),
                    "app": rec.get("app"),
                    "time": rec.get("time"),
                    "writer": rec.get("writer"),
                    "flags": flags(rec),
                    "drop": bool(d.get("drop")),
                    "relabel": d.get("label"),
                }
            )
        return {
            "inputs": [str(p) for p in self.inputs],
            "output": str(self.output),
            "groups": groups,
            "battuta": list(BATTUTA_CLASSES),
            "broken": self.broken,
            "repeats": self.repeats,
            "replies": replies,
        }

    def save(self, by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Take the page's decisions (by reply index), write the output and the review."""
        decisions: dict[str, dict[str, Any]] = {}
        for i, d in by_id.items():
            r = self.replies[int(i)]
            label = d.get("label")
            if label is not None and label not in CATALOG:
                raise ValueError(f"{label!r} is not a class")
            entry: dict[str, Any] = {}
            if d.get("drop"):
                entry["drop"] = True
            if label and label != r.record.get("label"):
                entry["label"] = label
            if entry:
                decisions[r.key] = entry
        lines, counts = cleaned_lines(self.replies, decisions)
        _write(self.output, "".join(line + "\n" for line in lines))
        review = {"inputs": [str(p) for p in self.inputs], "decisions": decisions}
        _write(self.review, json.dumps(review, indent=1))
        self.decisions = decisions
        return {**counts, "output": str(self.output)}


def serve(review: Review, port: int, open_browser: bool) -> None:
    hosts: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _local(self) -> bool:
            # Another site in the browser, or a name rebound to 127.0.0.1, gets nothing
            if self.headers.get("Host") in hosts:
                return True
            self._send(403, b"forbidden", "text/plain")
            return False

        def do_GET(self) -> None:
            if not self._local():
                return
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/replies":
                self._send(200, json.dumps(review.payload()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self._local():
                return
            if self.path != "/save" or self.headers.get("Content-Type") != "application/json":
                self._send(404, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                result = review.save(body.get("decisions", {}))
            except (ValueError, KeyError, IndexError) as e:
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
                return
            self._send(200, json.dumps(result).encode(), "application/json")

        def log_message(self, format: str, *args: object) -> None:  # quiet
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # the port is taken
    bound = server.server_address[1]
    hosts.update({f"127.0.0.1:{bound}", f"localhost:{bound}"})
    url = f"http://127.0.0.1:{bound}/"
    print(
        f"{len(review.replies)} replies from {len(review.inputs)} file(s)"
        f" ({review.repeats} repeats left out, {len(review.broken)} lines unreadable);"
        f" saving to {review.output}\nopen {url} (Ctrl+C to stop)",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("replies", type=Path, nargs="+", help="shells' replies.jsonl files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="the cleaned file (default: replies_clean.jsonl next to the first input)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="only print the address")
    args = parser.parse_args()
    output = args.output or args.replies[0].with_name("replies_clean.jsonl")
    if output.resolve() in {p.resolve() for p in args.replies}:
        raise SystemExit(f"{output} is an input: the inputs are never written")
    serve(Review(args.replies, output), args.port, not args.no_browser)


if __name__ == "__main__":
    main()
