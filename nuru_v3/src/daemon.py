"""
Daemon principal de NURU V3.

Point d'entrée unique pour l'application.
Gère:
- Initialisation asynchrone
- Logging structuré
- Gestion signaux (SIGINT, SIGTERM)
- Boucle principale asyncio
"""

import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import config
from src.nuru_core import get_nuru


def setup_logging() -> None:
    """Configurer le logging avec rotation."""
    # Handlers
    log_file = config.LOGS_DIR / "nuru.log"

    # File handler avec rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.LOG_LEVEL))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Désactiver logs trop verbeux
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main_async():
    """Boucle principale async."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("NURU V3 - Assistant IA Personnel")
    logger.info("=" * 60)
    logger.info(f"Plateforme: macOS Apple Silicon")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info("=" * 60)

    # Initialiser Nuru
    nuru = await get_nuru()

    # Gérer signaux
    loop = asyncio.get_event_loop()

    def signal_handler(sig):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        asyncio.create_task(shutdown(nuru))

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    # Boucle interactive simple
    logger.info("NURU prêt. Tapez vos questions (Ctrl+C pour quitter).")
    print("\n\033[96mNURU V3 prêt. Posez votre question:\033[0m")

    try:
        while True:
            # Lire input utilisateur
            query = await asyncio.get_event_loop().run_in_executor(
                None,
                input,
                "> ",
            )

            if not query.strip():
                continue

            if query.strip().lower() in ["quit", "exit", "q"]:
                break

            # Traiter requête
            print("\n\033[95mNURU:\033[0m ", end="", flush=True)

            async for token in nuru.process_query(query.strip()):
                print(token, end="", flush=True)

            print("\n")

    except EOFError:
        pass

    finally:
        await shutdown(nuru)


async def shutdown(nuru):
    """Arrêt propre."""
    logger = logging.getLogger(__name__)
    logger.info("Fermeture de NURU...")

    await nuru.shutdown()

    logger.info("NURU arrêté.")
    sys.exit(0)


def main():
    """Point d'entrée principal."""
    setup_logging()

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.getLogger(__name__).error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
