#!/usr/bin/env python3
"""Local, read-only browser for the large train.parquet dataset."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "parquet_inspector_web"
PARQUET = ROOT / "train.parquet"
SAFE_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return scalar(value.tolist())
    return str(value)


class Inspector:
    def __init__(self, parquet: Path):
        self.parquet = parquet.resolve()
        self.db = duckdb.connect()
        self.source = "read_parquet(?)"
        self.columns = [
            {"name": row[0], "type": row[1]}
            for row in self.db.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(self.parquet)]
            ).fetchall()
        ]
        self.names = {c["name"] for c in self.columns}

    def _column(self, name: str) -> str:
        if name not in self.names or not SAFE_COLUMN.match(name):
            raise ValueError(f"Neplatný sloupec: {name}")
        return f'"{name}"'

    def info(self):
        count = self.db.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(self.parquet)]
        ).fetchone()[0]
        return {
            "file": self.parquet.name,
            "size": self.parquet.stat().st_size,
            "rows": count,
            "columns": self.columns,
        }

    def rows(self, query):
        page = max(1, int(query.get("page", ["1"])[0]))
        limit = min(200, max(10, int(query.get("limit", ["50"])[0])))
        offset = (page - 1) * limit
        sort = query.get("sort", [""])[0]
        direction = "DESC" if query.get("direction", ["asc"])[0] == "desc" else "ASC"
        search = query.get("search", [""])[0].strip()
        field = query.get("field", [""])[0]

        where, params = [], [str(self.parquet)]
        if search:
            searchable = [
                c["name"] for c in self.columns
                if any(t in c["type"].upper() for t in ("VARCHAR", "DOUBLE", "FLOAT", "INT"))
            ]
            if field:
                searchable = [field] if field in self.names else []
            if not searchable:
                raise ValueError("Pro hledání není vybraný platný sloupec.")
            where.append("(" + " OR ".join(
                f"CAST({self._column(c)} AS VARCHAR) ILIKE ?" for c in searchable
            ) + ")")
            params.extend([f"%{search}%"] * len(searchable))
        clause = " WHERE " + " AND ".join(where) if where else ""
        order = f" ORDER BY {self._column(sort)} {direction} NULLS LAST" if sort else ""
        sql = f"SELECT * FROM read_parquet(?){clause}{order} LIMIT {limit} OFFSET {offset}"
        cursor = self.db.execute(sql, params)
        names = [d[0] for d in cursor.description]
        result = [{k: scalar(v) for k, v in zip(names, row)} for row in cursor.fetchall()]
        return {"page": page, "limit": limit, "rows": result, "has_more": len(result) == limit}

    def distinct(self, query):
        column = self._column(query.get("column", [""])[0])
        values = self.db.execute(
            f"SELECT {column}, count(*) AS n FROM read_parquet(?) "
            f"GROUP BY {column} ORDER BY n DESC NULLS LAST LIMIT 50",
            [str(self.parquet)],
        ).fetchall()
        return [{"value": scalar(v), "count": n} for v, n in values]


class Handler(BaseHTTPRequestHandler):
    inspector: Inspector

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/info":
                return self.send_json(self.inspector.info())
            if parsed.path == "/api/rows":
                return self.send_json(self.inspector.rows(parse_qs(parsed.query)))
            if parsed.path == "/api/distinct":
                return self.send_json(self.inspector.distinct(parse_qs(parsed.query)))
            relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (STATIC / relative).resolve()
            if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                raise FileNotFoundError
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Lokální prohlížeč train.parquet")
    parser.add_argument("--file", type=Path, default=PARQUET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.file.exists():
        parser.error(f"Soubor neexistuje: {args.file}")
    Handler.inspector = Inspector(args.file)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Parquet Inspector běží na http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nUkončuji.")


if __name__ == "__main__":
    main()
