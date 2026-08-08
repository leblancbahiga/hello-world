"""Point d'entrée : `python -m nuru` ou la commande `nuru`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import uvicorn

from nuru import __version__
from nuru.core import NuruApp
from nuru.meta import rss_mb
from nuru.settings import default_home


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nuru", description="NURU Lite — assistant IA personnel léger")
    parser.add_argument("--home", type=str, default=None, help="répertoire de données (défaut: ~/.nuru)")
    parser.add_argument("--host", type=str, default=None, help="adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (défaut: 8734)")
    parser.add_argument("--timing", action="store_true", help="affiche les temps de démarrage")
    parser.add_argument("--log-level", type=str, default="warning", help="niveau de log (debug/info/warning)")
    parser.add_argument("--version", action="version", version=f"nuru-lite {__version__}")
    return parser


async def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    home = Path(args.home).expanduser() if args.home else default_home()
    app = NuruApp(home)

    t0 = time.perf_counter()
    await app.start()
    t1 = time.perf_counter()

    host = args.host or str(app.settings.get("server.host", "127.0.0.1"))
    port = args.port or int(app.settings.get("server.port", 8734))

    if args.timing:
        print(f"[nuru] boot en {(t1 - t0) * 1000:.0f} ms | RAM {rss_mb():.0f} Mo | home {home}", flush=True)
    print(f"[nuru] {app.version} prêt — http://{host}:{port}", flush=True)

    config = uvicorn.Config(
        app.server.fastapi_app,
        host=host,
        port=port,
        log_level=args.log_level,
        access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        await app.shutdown()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
