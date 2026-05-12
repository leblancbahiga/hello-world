"""
Ingestion de documents pour RAG.

Supporte:
- TXT, MD
- PDF (via pypdf)
- Découpage intelligent en chunks
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from src.config import config
from src.rag_engine import RAGEngine, Chunk


logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Ingère des documents dans le moteur RAG.

    Pipeline:
    1. Lecture fichier
    2. Nettoyage texte
    3. Chunking sémantique
    4. Embedding
    5. Stockage vectoriel
    """

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    async def ingest(self, file_path: str) -> int:
        """
        Ingérer un fichier complet.

        Args:
            file_path: Chemin vers le fichier

        Returns:
            Nombre de chunks ingérés
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        logger.info(f"Ingestion de {path.name}...")

        # Lire selon extension
        text = await self._read_file(path)

        if not text.strip():
            logger.warning("Fichier vide")
            return 0

        # Chunking
        chunks = self._chunk_text(text, path.name)

        # Embedding et stockage
        count = await self._store_chunks(chunks)

        logger.info(f"{count} chunks ingérés depuis {path.name}")
        return count

    async def _read_file(self, path: Path) -> str:
        """Lire un fichier selon son extension."""
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md", ".markdown"]:
            return path.read_text(encoding="utf-8")

        elif suffix == ".pdf":
            return await self._read_pdf(path)

        else:
            raise ValueError(f"Extension non supportée: {suffix}")

    async def _read_pdf(self, path: Path) -> str:
        """Extraire texte d'un PDF."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            text_parts = []

            for page in reader.pages:
                text_parts.append(page.extract_text())

            return "\n\n".join(text_parts)

        except ImportError:
            logger.error("pypdf non installé. pip install pypdf")
            raise

        except Exception as e:
            logger.error(f"Erreur lecture PDF: {e}")
            raise

    def _chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Découper un texte en chunks sémantiques.

        Stratégie:
        - Découpage par paragraphes
        - Regroupement jusqu'à ~400 tokens
        - Overlap de 50 tokens entre chunks
        """
        chunks: List[Chunk] = []

        # Nettoyer texte
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Séparer par paragraphes
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = []
        current_length = 0
        target_length = config.CHUNK_SIZE * 4  # ~4 chars par token
        overlap_length = config.CHUNK_OVERLAP * 4

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            # Si paragraphe trop long, le découper
            if para_tokens > config.CHUNK_SIZE:
                # Sauvegarder chunk actuel si existe
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, source, len(chunks)))
                    current_chunk = []
                    current_length = 0

                # Découper paragraphe en sous-chunks
                sub_chunks = self._split_paragraph(para, config.CHUNK_SIZE)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk(sub, source, len(chunks)))

            # Ajouter au chunk courant
            elif current_length + para_tokens <= target_length:
                current_chunk.append(para)
                current_length += para_tokens

            else:
                # Chunk plein, créer nouveau
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, source, len(chunks)))

                # Garder overlap
                overlap_chars = overlap_length
                if len(current_chunk[-1]) > overlap_chars:
                    current_chunk = [current_chunk[-1][-overlap_chars:]]
                    current_length = self._count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

                current_chunk.append(para)
                current_length += para_tokens

        # Dernier chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, source, len(chunks)))

        logger.debug(f"{len(chunks)} chunks créés")
        return chunks

    def _split_paragraph(self, paragraph: str, max_tokens: int) -> List[str]:
        """Découper un paragraphe long en sous-parties."""
        sentences = paragraph.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sent_len = self._count_tokens(sentence)

            if current_len + sent_len <= max_tokens:
                current.append(sentence)
                current_len += sent_len
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
                current_len = sent_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Estimer nombre de tokens (approximation)."""
        # Approximation: 1 token ≈ 4 caractères en français
        return len(text) // 4

    def _create_chunk(self, text: str, source: str, index: int) -> Chunk:
        """Créer un chunk avec ID unique."""
        chunk_id = hashlib.md5(f"{source}:{index}:{text[:50]}".encode()).hexdigest()

        return Chunk(
            id=chunk_id,
            text=text,
            source=source,
        )

    async def _store_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embedder et stocker les chunks.

        Retourne le nombre de chunks stockés.
        """
        if not chunks:
            return 0

        # Batch embeddings pour efficacité
        texts = [c.text for c in chunks]
        embeddings = await self.rag_engine.embedder.embed(texts)

        # Assigner embeddings aux chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            await self.rag_engine.add_chunk(chunk)

        return len(chunks)
